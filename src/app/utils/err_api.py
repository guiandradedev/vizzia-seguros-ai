class ErrAPI(Exception):
    def __init__(self, mensagem="O valor fornecido é inválido", status_code=400):
        self.mensagem = mensagem
        self.status_code = status_code
        super().__init__(self.mensagem)

    def __str__(self):
        return f'{self.mensagem}'