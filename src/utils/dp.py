"""Differential Privacy engine for SparsePay-RAG.

Implements zCDP accounting for the exponential mechanism used in
DP Contrastive Decoding (Sec. 3.4 of the paper).

Based on the InvisibleInk-style mechanism:
  rho_token = 2 * C^2 / (K^2 * tau^2)
"""

import math
import torch
import torch.nn.functional as F


class DPExpenseOverflow(Exception):
    """Raised when the privacy budget is exhausted."""
    pass


class ClippedLogitsDP:
    """DP engine for logits-based contrastive decoding with zCDP accounting.

    All budget tracking is done in rho-space (zCDP).
    rho is additive under composition: rho_total = rho_cls + rho_gen.
    The clipping norm is derived directly from rho_per_token.

    Uses the exponential mechanism via softmax sampling.
    """

    def __init__(
        self,
        rho_per_token: float,
        target_rho: float,
        target_delta: float,
        num_private_models: int,
        temperature: float,
        fail_mode: str = 'stop',
    ):
        """
        Args:
            rho_per_token: zCDP rho for a single DP token generation.
            target_rho: total rho budget for the entire query (generation side).
            target_delta: total delta budget for the entire query.
            num_private_models: K (number of retrieved documents).
            temperature: sampling temperature tau.
            fail_mode: 'stop' raises DPExpenseOverflow; 'fallback' returns None.
        """
        self.rho_per_token = rho_per_token
        self.target_rho = target_rho
        self.target_delta = target_delta
        self.num_private = num_private_models
        self.temperature = temperature
        self.fail_mode = fail_mode

        self.tokens_generated = 0
        self.budget_exhausted = False

        # Compute clipping norm directly from rho_per_token:
        # rho_token = 2 * C^2 / (K^2 * tau^2)  =>  C = K * tau * sqrt(rho_token / 2)
        self.clip_norm = self._compute_clip_norm()

    # ------------------------------------------------------------------
    #  Clipping norm derivation (Appendix B.5 of the paper)
    # ------------------------------------------------------------------

    def _compute_clip_norm(self) -> float:
        """Derive clipping norm C from rho_per_token.

        From the paper:  rho_token = 2 * C^2 / (K^2 * tau^2)
        =>  C = K * tau * sqrt(rho_token / 2)
        """
        clip_norm = math.sqrt(0.5 * self.rho_per_token) * float(self.num_private) * float(self.temperature)
        return clip_norm

    # ------------------------------------------------------------------
    #  zCDP <-> (eps, delta)-DP conversion utilities
    # ------------------------------------------------------------------

    def _cdp_rho(self, eps: float, delta: float) -> float:
        """Compute smallest rho such that rho-zCDP implies (eps, delta)-DP."""
        if eps == 0.0:
            return 0.0
        rhomin, rhomax = 0.0, max(1.0, eps + 1.0)
        for _ in range(2000):
            rho = 0.5 * (rhomin + rhomax)
            if self._cdp_delta(rho, eps) <= delta:
                rhomin = rho
            else:
                rhomax = rho
        return float(rhomin)

    def _cdp_delta(self, rho: float, eps: float) -> float:
        """Convert rho-zCDP to delta for a given eps.

        Uses binary search over alpha to find the optimal conversion.
        """
        if rho == 0:
            return 0.0
        amin, amax = 1.0001, max(2.0, (eps + 1.0) / (2.0 * rho) + 2.0)
        for _ in range(1000):
            alpha = 0.5 * (amin + amax)
            derivative = (2.0 * alpha - 1.0) * rho - eps + math.log1p(-1.0 / alpha)
            if derivative < 0:
                amin = alpha
            else:
                amax = alpha
        alpha = 0.5 * (amin + amax)
        exponent = (alpha - 1.0) * (alpha * rho - eps) + alpha * math.log1p(-1.0 / alpha)
        try:
            delta = math.exp(exponent) / (alpha - 1.0)
        except OverflowError:
            delta = 0.0
        return min(max(delta, 0.0), 1.0)

    def _cdp_eps(self, rho: float, delta: float) -> float:
        """Compute smallest eps such that rho-zCDP implies (eps, delta)-DP."""
        if rho == 0:
            return 0.0
        epsmin, epsmax = 0.0, rho + 2.0 * math.sqrt(max(0, rho * math.log(1.0 / delta)))
        for _ in range(1000):
            eps = 0.5 * (epsmin + epsmax)
            if self._cdp_delta(rho, eps) <= delta:
                epsmax = eps
            else:
                epsmin = eps
        return float(epsmax)

    @staticmethod
    def _cdp_eps_static(rho: float, delta: float) -> float:
        """Static wrapper for _cdp_eps (no instance needed)."""
        helper = ClippedLogitsDP(0, 0, 0, 0, 1, 1)
        return helper._cdp_eps(rho, delta)

    # ------------------------------------------------------------------
    #  Budget tracking
    # ------------------------------------------------------------------

    def compute_rho(self, num_toks: int) -> float:
        """Compute cumulative zCDP rho for `num_toks` generated tokens.

        Since rho_per_token is set at init, this is simply:
          rho_tot = num_toks * rho_per_token
        """
        return float(num_toks) * self.rho_per_token

    def get_dp_expense(self) -> tuple[float, float]:
        """Return current (epsilon, delta) privacy expenditure.

        Internally tracks rho; converts to eps only for external reporting.
        """
        rho = self.compute_rho(self.tokens_generated)
        eps = self._cdp_eps(rho, self.target_delta)
        return eps, self.target_delta

    def get_rho_expense(self) -> float:
        """Return current rho expenditure (native zCDP metric)."""
        return self.compute_rho(self.tokens_generated)

    def check_budget(self) -> None:
        """Check if the privacy budget is exhausted; raise if so.

        Comparison is in rho-space: current_rho > target_rho => exhausted.
        """
        if self.budget_exhausted:
            raise DPExpenseOverflow()
        current_rho = self.compute_rho(self.tokens_generated)
        if current_rho > self.target_rho:
            self.budget_exhausted = True
            raise DPExpenseOverflow()

    # ------------------------------------------------------------------
    #  Per-step operations
    # ------------------------------------------------------------------

    def clip_and_average(self, pri_logits: torch.Tensor) -> torch.Tensor:
        """Clip per-sample logits and average.

        Used by Logits Average (LA) decoding; not the primary CD path.

        Args:
            pri_logits: [num_private, vocab_size]

        Returns:
            Averaged logits [vocab_size]
        """
        z_max = torch.max(pri_logits, dim=-1, keepdim=True).values
        clipped = torch.clamp(pri_logits - z_max + self.clip_norm, min=-self.clip_norm)
        return clipped.mean(dim=0)

    def sample(self, avg_logits: torch.Tensor) -> torch.Tensor:
        """Exponential mechanism sampling with temperature.

        Args:
            avg_logits: utility vector of shape [vocab_size]

        Returns:
            Sampled token id (scalar tensor).
        """
        scaled = avg_logits / self.temperature
        probs = F.softmax(scaled, dim=-1)
        token = torch.multinomial(probs, num_samples=1).squeeze(-1)
        self.tokens_generated += 1
        return token

    # ------------------------------------------------------------------
    #  Static helper for per-query budget allocation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_per_query_budget(
        total_rho: float, num_compositions: int
    ) -> float:
        """Allocate per-step rho from total rho budget via zCDP composition.

        Since zCDP composition is linear in rho:
          rho_per_step = rho_total / num_compositions

        Args:
            total_rho: Total rho budget (zCDP).
            num_compositions: Number of sequential compositions.

        Returns:
            rho_per_step (float): rho budget per composition step.
        """
        return total_rho / float(num_compositions)

    @staticmethod
    def compute_cluster_noise_sigma(
        total_rho: float,
        budget_ratio: float = 0.2,
    ) -> tuple[float, float]:
        """Compute sigma for Gaussian noise on cluster sizes (Sec. 3.2 / Sec. 4).

        The paper fixes the budget split at 1:4, applied in rho-space
        (zCDP composition is linear in rho, not in epsilon).

        From the privacy analysis (Sec. 4):
          rho_total = rho_cls + rho_gen              (zCDP sequential composition)
          rho_cls   = budget_ratio * rho_total       (1/5 for 1:4 split)
          rho_cls   = 1 / (2 * sigma^2)             (Gaussian mechanism, l2-sens = 1)

        Arguments:
            total_rho: total rho budget for the full pipeline (zCDP).
            budget_ratio: fraction of rho_total allocated to cluster noise
                          (default 0.2 for the 1:4 rho-space split).

        Returns:
            (sigma, rho_cls): sigma for Gaussian noise and rho_cls budget used.
        """
        # rho-space split: rho_cls = budget_ratio * rho_total
        rho_cls = budget_ratio * total_rho

        # sigma from rho_cls = 1 / (2 * sigma^2)
        if rho_cls <= 1e-15:
            sigma = math.sqrt(0.5 / 1e-15)  # large sigma -> near-zero privacy cost
        else:
            sigma = math.sqrt(0.5 / rho_cls)

        return sigma, rho_cls

    @staticmethod
    def compute_generation_rho(
        total_rho: float,
        budget_ratio: float = 0.2,
    ) -> float:
        """Compute the generation-side rho budget (Sec. 4).

        rho_total = rho_cls + rho_gen
        rho_cls   = budget_ratio * rho_total   (1/5)
        rho_gen   = (1 - budget_ratio) * rho_total  (4/5)

        Arguments:
            total_rho: total rho budget for the full pipeline (zCDP).
            budget_ratio: fraction of rho_total for cluster noise (default 0.2).

        Returns:
            generation_rho (float): rho budget available for DP contrastive decoding.
        """
        return (1.0 - budget_ratio) * total_rho