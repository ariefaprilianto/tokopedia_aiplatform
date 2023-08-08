from typing import Any, List
from vertexai.language_models import TextGenerationModel, TextGenerationResponse


class TextGenerationModelAIGateway(TextGenerationModel):

    def predict(
        self,
        prompts: str,
        *,
        max_output_tokens: int = TextGenerationModel._DEFAULT_MAX_OUTPUT_TOKENS,
        temperature: float = TextGenerationModel._DEFAULT_TEMPERATURE,
        top_k: int = TextGenerationModel._DEFAULT_TOP_K,
        top_p: float = TextGenerationModel._DEFAULT_TOP_P,
        **kwargs: Any,
    ) -> "TextGenerationResponse":
        """Gets model response for a single prompt.

        Args:
            prompt: Question to ask the model.
            max_output_tokens: Max length of the output text in tokens.
            temperature: Controls the randomness of predictions. Range: [0, 1].
            top_k: The number of highest probability vocabulary tokens to keep for top-k-filtering.
            top_p: The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling. Range: [0, 1].

        Returns:
            A `TextGenerationResponse` object that contains the text produced by the model.
        """

        # Access the 'request_id' argument from kwargs
        request_id = kwargs.get('request_id')
        api_key = kwargs.get('api_key')
        
        return self._batch_predict(
            prompts=[prompts],
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            request_id=request_id,
            api_key=api_key
        )[0]

    def _batch_predict(
        self,
        prompts: List[str],
        max_output_tokens: int = TextGenerationModel._DEFAULT_MAX_OUTPUT_TOKENS,
        temperature: float = TextGenerationModel._DEFAULT_TEMPERATURE,
        top_k: int = TextGenerationModel._DEFAULT_TOP_K,
        top_p: float = TextGenerationModel._DEFAULT_TOP_P,
        request_id: str = "",
        api_key: str = "",
    ) -> List["TextGenerationResponse"]:
        """Gets model response for a single prompt.

        Args:
            prompts: Questions to ask the model.
            max_output_tokens: Max length of the output text in tokens.
            temperature: Controls the randomness of predictions. Range: [0, 1].
            top_k: The number of highest probability vocabulary tokens to keep for top-k-filtering.
            top_p: The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling. Range: [0, 1].
            request_id: The parameter serves as a unique identifier for the API request. 
                        When provided, the AI Gateway can associate this identifier with the request and use it for various purposes, 
                        such as tracking token usage, logging, metering, and other relevant metrics.
                        By including a specific request_id, you enable the AI Gateway to perform detailed monitoring and analysis of the request, 
                        allowing for accurate tracking of resource consumption and other metrics associated with the API call.

        Returns:
            A list of `TextGenerationResponse` objects that contain the texts produced by the model.
        """
        instances = [{"content": str(prompt)} for prompt in prompts]
        prediction_parameters = {
            "temperature": temperature,
            "maxDecodeSteps": max_output_tokens,
            "topP": top_p,
            "topK": top_k,
            "request-id": request_id,
            "api-key": api_key,
        }

        prediction_response = self._endpoint.predict(
            instances=instances,
            parameters=prediction_parameters,
        )

        results = []
        for prediction in prediction_response.predictions:
            safety_attributes_dict = prediction.get("safetyAttributes", {})
            results.append(
                TextGenerationResponse(
                    text=prediction["content"],
                    _prediction_response=prediction_response,
                    is_blocked=safety_attributes_dict.get("blocked", False),
                    safety_attributes=dict(
                        zip(
                            safety_attributes_dict.get("categories", []),
                            safety_attributes_dict.get("scores", []),
                        )
                    ),
                )
            )
        return results