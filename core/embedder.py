from zai import ZhipuAiClient

class ZhipuEmbedder:
    def __init__(self,api_key:str):
        self.client = ZhipuAiClient(api_key = api_key)
        self.model = "embedding-3"

    def get_embedding(self,text:str):
        response = self.client.embeddings.create(
            model = self.model,
            input = text
        )
        return response.data[0].embedding

