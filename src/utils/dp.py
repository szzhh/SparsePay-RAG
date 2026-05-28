"""Differential Privacy engine for SparsePay-RAG.

Implements zCDP accounting for the exponential mechanism used in
DP Contrastive Decoding (Sec. 3.4 of the paper).

Based on the InvisibleInk-style mechanism:
  rho_token = C^2 / (2 * K^2 * tau^2)
"""

import math
import torch
import torch.nn.functional as F


class DPExpenseOverflow(Exception):
    """Raised when the privacy budget is exhausted."""
    pass


class ClippedLogitsDP:
    """DP engine for logits-based contrastive decoding with zCDP accounting.

    Automatically computes the clipping norm from the per-token privacy budget.
    Uses the exponential mechanism via softmax sampling.
    """

    def __init__(
        self,
        eps_per_token: float,
        delta_per_token: float,
        target_eps: float,
        target_delta: float,
        num_private_models: int,
        temperature: float,
        fail_mode: str = 'stop',
    ):
        """
        Args:
            eps_per_token: epsilon for a single DP token generation.
            delta_per_token: delta for a single DP token generation.
            target_eps: total epsilon budget for the entire query.
            target_delta: total delta budget for the entire query.
            num_private_models: K (number of retrieved documents).
            temperature: sampling temperature tau.
            fail_mode: 'stop' raises DPExpenseOverflow; 'fallback' returns None.
        """
        self.eps_per_token = eps_per_token
        self.delta_per_token = delta_per_token
        self.target_eps = target_eps
        self.target_delta = target_delta
        self.num_private = num_private_models
        self.temperature = temperature
        self.fail_mode = fail_mode

        self.tokens_generated = 0
        self.budget_exhausted = False

        # Compute clipping norm from per-token epsilon
        self.clip_norm = self._compute_clip_norm()

    # ------------------------------------------------------------------
    #  Clipping norm derivation (Appendix B.5 of the paper)
    # ------------------------------------------------------------------

    def _compute_clip_norm(self) -> float:
        """Derive clipping norm C from per-token privacy budget.

        From the paper:  rho_token = C^2 / (2 * K^2 * tau^2)
        =>  C = K * tau * sqrt(2 * rho_token)
        """
        rho_tok = self._cdp_rho(self.eps_per_token, self.delta_per_token)
        clip_norm = math.sqrt(2.0 * rho_tok) * float(self.num_private) * float(self.temperature)
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

        rho_tok = 0.5 * (C / (K * tau))^2
        rho_tot = num_toks * rho_tok
        """
        rho_tok = 0.5 * (float(self.clip_norm) / (float(self.num_private) * float(self.temperature))) ** 2
        return float(num_toks) * rho_tok

    def get_dp_expense(self) -> tuple[float, float]:
        """Return current (epsilon, delta) privacy expenditure."""
        rho = self.compute_rho(self.tokens_generated)
        eps = self._cdp_eps(rho, self.target_delta)
        return eps, self.target_delta

    def check_budget(self) -> None:
        """Check if the privacy budget is exhausted; raise if so."""
        if self.budget_exhausted:
            raise DPExpenseOverflow()
        current_eps, _ = self.get_dp_expense()
        if current_eps > self.target_eps:
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
        total_eps: float, total_delta: float, num_compositions: int
    ) -> tuple[float, float]:
        """Allocate per-step (eps, delta) from total budget via zCDP composition.

        Steps:
          1. (total_eps, total_delta) -> rho_total  (exact conversion)
          2. rho_single = rho_total / num_compositions
          3. rho_single -> (target_eps, target_delta)

        Args:
            total_eps: Total epsilon budget.
            total_delta: Total delta budget.
            num_compositions: Number of sequential compositions.

        Returns:
            (eps_per_step, delta_per_step) tuple.
        """
        # Helper: create a temporary instance to use cdp conversion methods
        helper = ClippedLogitsDP(0, 0, 0, 0, 1, 1)

        # Step 1: (total_eps, total_delta) -> rho_total
        rho_total = helper._cdp_rho(total_eps, total_delta)

        # Step 2: linear allocation
        rho_single = rho_total / float(num_compositions)

        # Step 3: rho_single -> eps_single
        eps_single = helper._cdp_eps(rho_single, total_delta)

        return eps_single, total_delta

    @staticmethod
    def compute_cluster_noise_sigma(
        total_eps: float,
        total_delta: float,
        budget_ratio: float = 0.2,
    ) -> tuple[float, float]:
        """Compute sigma for Gaussian noise on cluster sizes (Sec. 3.2 / Sec. 4).

        The paper fixes the budget split at 1:4, applied **in rho-space**
        (zCDP composition is linear in rho, not in epsilon).

        From the privacy analysis (Sec. 4):
          rho_total = rho_cls + rho_gen              (zCDP sequential composition)
          rho_cls   = budget_ratio * rho_total       (1/5 for 1:4 split)
          rho_cls   = 1 / (2 * sigma^2)             (Gaussian mechanism, l2-sens = 1)

        Arguments:
            total_eps: total epsilon budget for the full pipeline.
            total_delta: total delta budget.
            budget_ratio: fraction of **rho_total** allocated to cluster noise
                          (default 0.2 for the 1:4 rho-space split).

        Returns:
            (sigma, rho_cls): sigma for Gaussian noise and rho_cls budget used.
        """
        # 1. Convert total_eps -> rho_total (exact binary search)
        helper = ClippedLogitsDP(0, 0, 0, 0, 1, 1)
        rho_total = helper._cdp_rho(total_eps, total_delta)

        # 2. rho-space split: rho_cls = budget_ratio * rho_total
        rho_cls = budget_ratio * rho_total

        # 3. sigma from rho_cls = 1 / (2 * sigma^2)
        if rho_cls <= 1e-15:
            sigma = math.sqrt(0.5 / 1e-15)  # large sigma -> near-zero privacy cost
        else:
            sigma = math.sqrt(0.5 / rho_cls)

        return sigma, rho_cls

    @staticmethod
    def compute_generation_rho(
        total_eps: float,
        total_delta: float,
        budget_ratio: float = 0.2,
    ) -> float:
        """Compute the generation-side rho budget (Sec. 4).

        The paper fixes the budget split at 1:4 in **rho-space**:
          rho_total = rho_cls + rho_gen
          rho_cls   = budget_ratio * rho_total   (1/5)
          rho_gen   = (1 - budget_ratio) * rho_total  (4/5)

        Arguments:
            total_eps: total epsilon budget for the full pipeline.
            total_delta: total delta budget.
            budget_ratio: fraction of rho_total for cluster noise (default 0.2).

        Returns:
            generation_rho (float): rho budget available for DP contrastive decoding.
        """
        helper = ClippedLogitsDP(0, 0, 0, 0, 1, 1)
        rho_total = helper._cdp_rho(total_eps, total_delta)
        return (1.0 - budget_ratio) * rho_total