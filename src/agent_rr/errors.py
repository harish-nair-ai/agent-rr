"""The single user-facing error type: rendered as `error: <message>` with a documented exit code."""


class AgentRRError(Exception):
    def __init__(self, message: str, exit_code: int = 2) -> None:
        super().__init__(message)
        self.exit_code = exit_code
