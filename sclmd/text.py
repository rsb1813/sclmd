# A reply's tokens as text while it is being written (the chat and the server).


class Pieces:
    """A reply's tokens as pieces of text, thinking and answer apart: add(token) -> (kind, text) or None."""

    def __init__(self, tok, stop: tuple, opening: int, closing: int):
        self.tok, self.stop, self.opening, self.closing = tok, stop, opening, closing
        self.kind, self.ids, self.done, self.shown, self.name = "answer", [], 0, 0, False

    def add(self, token: int):
        if token in self.stop:
            return None
        if token == self.opening or token == self.closing:
            self.kind, self.ids, self.done, self.shown, self.name = ("think" if token == self.opening else "answer"), [], 0, 0, token == self.opening
            return None
        self.ids.append(token)
        # the text of the tokens from `done` on, less what of it was shown: a few tokens are decoded, not the reply
        before = self.tok.decode(self.ids[self.done : self.shown], skip_special_tokens=False)
        text = self.tok.decode(self.ids[self.done :], skip_special_tokens=False)
        if text.endswith("\ufffd") or len(text) <= len(before):  # a character is still missing bytes
            return None
        piece = text[len(before):]
        self.done, self.shown = self.shown, len(self.ids)
        if self.name:  # the thinking channel's name comes first: "thought\n"
            whole = self.tok.decode(self.ids, skip_special_tokens=False)
            if "thought\n".startswith(whole):
                return None
            self.name = False
            if whole.startswith("thought\n"):
                piece = whole[len("thought\n"):]
        return (self.kind, piece) if piece else None
