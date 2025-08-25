from typing import Callable, Awaitable, Tuple, List, Dict, Optional


class Provider:
    """Abstract interface for an LLM provider.

    Implementations must provide an async call method and a set_model method.
    """

    async def call(
        self,
        messages: List[Dict[str, str]],
        pdf_path: Optional[str],
        *,
        temperature: float = 0.7,
    ) -> Tuple[str, int, int, str]:
        raise NotImplementedError

    def set_model(self, label_or_id: str) -> str:
        raise NotImplementedError


class ModuleProvider(Provider):
    """Adapter that wraps module-level call/set_model functions into a Provider."""

    def __init__(
        self,
        call_func: Callable[[List[Dict[str, str]], Optional[str]], Awaitable[Tuple[str, int, int, str]]],
        set_model_func: Callable[[str], str],
    ):
        self._call_func = call_func
        self._set_model_func = set_model_func

    async def call(
        self,
        messages: List[Dict[str, str]],
        pdf_path: Optional[str],
        *,
        temperature: float = 0.7,
    ) -> Tuple[str, int, int, str]:
        # Delegate temperature via keyword to maintain signature compatibility
        return await self._call_func(messages, pdf_path, temperature=temperature)

    def set_model(self, label_or_id: str) -> str:
        return self._set_model_func(label_or_id)


