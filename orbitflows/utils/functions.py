import torch


def gamma(x : torch.tensor):
    return torch.special.gammaln(x).exp()


def hyp1f1(a, b, z, max_iter=100, tol=1e-8):
    # Ensure tensors broadcast
    a, b, z = map(lambda t: t.unsqueeze(-1), (a, b, z))
    k = torch.arange(0, max_iter, device=z.device).float()
    # Compute (a)_k / (b)_k for each k
    num = torch.lgamma(a + k) - torch.lgamma(a)
    den = torch.lgamma(b + k) - torch.lgamma(b)
    poch_ratio = torch.exp(num - den)
    # k! via gammaln
    fact = torch.exp(torch.lgamma(k + 1))
    term = poch_ratio * (z**k) / fact  # shape (..., max_iter)
    series = torch.cumsum(term, dim=-1)
    # Early stopping
    last_terms = term[..., -1]
    if torch.max(last_terms.abs()) < tol:
        return series[..., -1]
    else:
        return series[..., -1]


# def hyp1f1(a: torch.tensor, b: torch.tensor, x: torch.tensor):
#   r"""The 1F1 hypergeometric function.

#   JAX implementation of :obj:`scipy.special.hyp1f1`.

#   .. math::

#      \mathrm{hyp1f1}(a, b, x) = {}_1F_1(x;a, b) = \sum_{k=0}^\infty \frac{(a)_k}{(b)_kk!}x^k

#   where :math:`(\cdot)_k` is the Pochammer symbol (refer to :func:`~jax.scipy.special.poch`).

#   The JAX version only accepts positive and real inputs. Values of ``a``, ``b``,
#   and ``x``, leading to high values of 1F1 may lead to erroneous results;
#   consider enabling double precision in this case. The convention for
#   ``a = b = 0`` is ``1``, unlike in scipy's implementation.

#   Args:
#     a: arraylike, real-valued
#     b: arraylike, real-valued
#     x: arraylike, real-valued

#   Returns:
#     array of 1F1 values.
#   """
#   # This is backed by https://doi.org/10.48550/arXiv.1407.7786
#   # There is room for improvement in the implementation using recursion to
#   # evaluate lower values of hyp1f1 when a or b or both are > 60-80
#   a, b, x = promote_args_inexact('hyp1f1', a, b, x)

#   result = lax.cond(lax.abs(x) < 100, _hyp1f1_serie, _hyp1f1_asymptotic, a, b, x)
#   index = (a == 0) * 1 + ((a == b) & (a != 0)) * 2 + ((b == 0) & (a != 0)) * 3

#   return lax.select_n(index,
#                       result,
#                       jnp.array(1, dtype=x.dtype),
#                       jnp.exp(x),
#                       jnp.array(np.inf, dtype=x.dtype))