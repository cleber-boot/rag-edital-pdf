"""
gemini_retry.py — Utilitário centralizado de retry com backoff exponencial + jitter
                   para chamadas à API do Gemini (erros 429 / RESOURCE_EXHAUSTED).

Uso:
    from gemini_retry import gemini_retry

    @gemini_retry()
    def minha_chamada():
        return cliente.models.generate_content(...)
"""

import time
import random
import logging
import functools

logger = logging.getLogger(__name__)

# ── limites do plano gratuito Gemini AI Studio ──────────────────────────────
# gemini-2.5-flash-lite: 15 req/min
# gemini-embedding-001 : 100 req/min
# Ambos disparam 429 quando ultrapassados.

_ERROS_429 = ("429", "RESOURCE_EXHAUSTED", "quota", "rate limit")


def _eh_429(excecao: Exception) -> bool:
    msg = str(excecao).lower()
    return any(t in msg for t in _ERROS_429)


def gemini_retry(
    max_tentativas: int = 7,
    espera_base: float = 10.0,
    espera_max: float = 120.0,
    jitter: float = 3.0,
):
    """
    Decorador que envolve qualquer chamada à API do Gemini com:
      - Detecção de erro 429 / RESOURCE_EXHAUSTED
      - Backoff exponencial: espera_base * 2^(tentativa-1)
      - Jitter aleatório para evitar thundering herd
      - Limite de espera em espera_max segundos

    Parâmetros
    ----------
    max_tentativas : int
        Número máximo de tentativas antes de re-lançar a exceção.
    espera_base : float
        Tempo inicial de espera em segundos (dobra a cada falha).
    espera_max : float
        Teto máximo de espera entre tentativas.
    jitter : float
        Variação aleatória adicionada à espera (±jitter/2).
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for tentativa in range(1, max_tentativas + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if not _eh_429(e):
                        raise  # erros que não são 429 sobem imediatamente

                    if tentativa == max_tentativas:
                        logger.error(
                            "[gemini_retry] Limite de %d tentativas atingido. "
                            "Último erro: %s", max_tentativas, e
                        )
                        raise

                    espera = min(espera_base * (2 ** (tentativa - 1)), espera_max)
                    espera += random.uniform(-jitter / 2, jitter / 2)
                    espera = max(espera, 1.0)

                    logger.warning(
                        "[gemini_retry] 429 detectado em '%s' "
                        "(tentativa %d/%d). Aguardando %.1fs...",
                        func.__name__, tentativa, max_tentativas, espera,
                    )
                    time.sleep(espera)

        return wrapper
    return decorator


def embed_com_retry(
    cliente,
    model: str,
    contents: list,
    config,
    max_tentativas: int = 7,
    espera_base: float = 10.0,
):
    """
    Versão funcional (sem decorador) para chamadas de embedding com retry.
    Útil quando a função é chamada inline dentro de loops.
    """
    for tentativa in range(1, max_tentativas + 1):
        try:
            return cliente.models.embed_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            if not _eh_429(e):
                raise
            if tentativa == max_tentativas:
                raise

            espera = min(espera_base * (2 ** (tentativa - 1)), 120.0)
            espera += random.uniform(-1.5, 1.5)
            espera = max(espera, 1.0)
            logger.warning(
                "[embed_com_retry] 429 no embedding (tentativa %d/%d). "
                "Aguardando %.1fs...", tentativa, max_tentativas, espera,
            )
            time.sleep(espera)
