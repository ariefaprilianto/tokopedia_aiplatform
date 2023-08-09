
from typing import Any, List
from vertexai.language_models import TextEmbeddingModel, TextEmbedding

class TextEmbeddingModelAIGateway(TextEmbeddingModel):
        
    def get_embeddings(
            self, 
            texts: List[str],
            **kwargs: Any,
        ) -> List["TextEmbedding"]:
            # Access the 'request_id' argument from kwargs
            request_id = kwargs.get('request_id')
            api_key = kwargs.get('api_key')

            print(f"request_id: {request_id}")
            print(f"api_key: {api_key}")
            
            instances = [{"content": str(text)} for text in texts]

            prediction_parameters = {
                "request-id": request_id,
                "api-key": api_key,
            }
                
            prediction_response = self._endpoint.predict(
                parameters=prediction_parameters,
                instances=instances,
            )

            return [
                TextEmbedding(
                    values=prediction["embeddings"]["values"],
                    _prediction_response=prediction_response,
                )
                for prediction in prediction_response.predictions
            ]