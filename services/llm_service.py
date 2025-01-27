import google.generativeai as genai
from config import Settings

settings = Settings()


class LLMService:
    def __init__(self):
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def generate_response(self, query: str, search_results: list[dict]):

        context_text = "\n\n".join(
            [
                f"Source {i+1} ({result['url']}):\n{result['content']}"
                for i, result in enumerate(search_results)
            ]
        )

        full_prompt = f"""

        You are DelveAi Perplexity-clone, a knowledgeable search assistant trained by Perplexity AI. Your task is to provide a detailed and informative response to a user's query. Please follow these guidelines:

        1. Start with a clear and concise answer to the user's question.
        2. Provide additional context, explanations, or examples to enrich the response.
        3. Include relevant citations or sources where applicable.
        4. Organize the information with headings or bullet points for clarity.
        5. if the context hasve image or urls make sure to include them in the response for the better understanding of the user if necessary.
            Context from web search:{context_text}
            User Query: {query}

        """

        response = self.model.generate_content(full_prompt, stream=True)

        for chunk in response:
            yield chunk.text
