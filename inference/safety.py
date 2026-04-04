class ToxicityFilter:
    def __init__(self):
        # Basic list for demonstration purposes
        self.banned_words = ['fuck', 'shit', 'bitch', 'asshole', 'cunt', 'dick', 'bastard', 'slut']
        
    def is_toxic(self, text: str) -> bool:
        lower_text = text.lower()
        # exact word match heuristic
        words = lower_text.replace('.', ' ').replace(',', ' ').split()
        for word in words:
            if word in self.banned_words:
                return True
        return False

    def clean(self, text: str) -> str:
        if self.is_toxic(text):
            return "[Content removed by safety filter]"
        return text
