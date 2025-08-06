from llama_cpp import Llama

class LocalLLM:
    """Wrapper for a local GGUF-based Large Language Model."""
    
    def __init__(self, model_path: str, n_ctx: int = 2048, n_threads: int = 4):
        """
        Initialize the local LLM. 
        
        Args:
            model_path (str): Path to the GGUF model file
            n_ctx (int): Context size for the model
            n_threads (int): Number of threads to use for generation
        """
        try:
            self.llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                verbose=False
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load LLM model from {model_path}: {e}")
            
    def generate_response(self, query: str, context: str) -> str:
        """
        Generate a response from the LLM based on a query and context.
        
        Args:
            query (str): The user's question
            context (str): The context retrieved from documents
            
        Returns:
            str: The generated answer
        """
        prompt = f"""
        Instructions:
        - You are a helpful assistant that answers questions based on the provided context.
        - Your answers should be concise and directly address the user's question.
        - If the answer to the question is not present in the context, you MUST say "I could not find an answer to that in the documents."
        - Do not make up information or use any knowledge outside of the provided context.

        Context:
        ---
        {context}
        ---
        
        Question: {query}
        
        Answer:
        """
        
        try:
            response = self.llm(
                prompt,
                max_tokens=512,
                stop=["\n", "Question:", "User:"],
                echo=False
            )
            
            # Extract the text from the response
            answer = response['choices'][0]['text'].strip()
            return answer
            
        except Exception as e:
            print(f"Error during LLM generation: {e}")
            return "I encountered an error while generating a response."

