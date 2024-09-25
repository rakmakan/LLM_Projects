import onnxruntime_genai as og
import time

class PhiONNXModelHandler:
    def __init__(self, verbose=False, timings=False):
        """
        Initialize the phiONNXHandler by loading the ONNX model and tokenizer.
        """
        self.verbose = verbose
        self.timings = timings
        self.model = None
        self.tokenizer = None
        self.tokenizer_stream = None

    def load_model(self, model_path=None):
        """
        Load the ONNX model and tokenizer.
        """
        if self.verbose: 
            print("Loading model...")
        self.model_path = model_path

        self.model = og.Model(f'{self.model_path}')
        self.tokenizer = og.Tokenizer(self.model)
        self.tokenizer_stream = self.tokenizer.create_stream()

        if self.verbose: 
            print("Model and tokenizer loaded")
        return self.model
        
    def create_prompt(self, input_text: str, template: str = '<|user|>\n{input} <|end|>\n<|assistant|>'):
        """
        Create a prompt using the specified template and input text.
        """
        return template.format(input=input_text)

    def chat_mode(self, args):
        """
        Interactive chat mode where the user inputs prompts, and the model generates responses.
        """
        chat_template = '<|user|>\n{input} <|end|>\n<|assistant|>'
        print("Entering chat mode. Type 'exit' to quit.")

        while True:
            text = input("Input: ")
            if text.lower() == 'exit':
                print("Exiting chat mode.")
                break

            if not text:
                print("Error, input cannot be empty")
                continue

            prompt = self.create_prompt(text, template=chat_template)
            answer = self.generate_answer(prompt, args)

    def generate_answer(self, prompt: str, args):
        """
        Generate a response based on the prompt and search options passed as arguments.
        """
        # Create search options based on the provided arguments
        do_sample = args.get('do_sample', True)
        max_length = args.get('max_length', 100)
        min_length = args.get('min_length', 1)
        top_p = args.get('top_p', 0.9)
        top_k = args.get('top_k', 50)
        temperature = args.get('temperature', 1.0)
        repetition_penalty = args.get('repetition_penalty', 1.0)

        search_options = {
            'do_sample': do_sample,
            'max_length': max_length,
            'min_length': min_length,
            'top_p': top_p,
            'top_k': top_k,
            'temperature': temperature,
            'repetition_penalty': repetition_penalty
        }

        # Filter out options that are None (not provided)
        search_options = {k: v for k, v in search_options.items() if v is not None}

        if self.verbose: 
            print(f"Encoding prompt: {prompt}")
        
        input_tokens = self.tokenizer.encode(prompt)

        params = og.GeneratorParams(self.model)
        params.set_search_options(**search_options)
        params.input_ids = input_tokens

        generator = og.Generator(self.model, params)

        if self.verbose: 
            print("Generator created, starting generation...")

        # Timing initialization
        if self.timings:
            started_timestamp = time.time()
            first_token_timestamp = 0
            first = True
            new_tokens = []

        # Generate tokens
        response = []
        try:
            while not generator.is_done():
                generator.compute_logits()
                generator.generate_next_token()

                if self.timings and first:
                    first_token_timestamp = time.time()
                    first = False

                new_token = generator.get_next_tokens()[0]
                decoded_token = self.tokenizer_stream.decode(new_token)
                response.append(decoded_token)
                
                if self.timings:
                    new_tokens.append(new_token)

                print(decoded_token, end='', flush=True)  # Print the response token-by-token

        except KeyboardInterrupt:
            print("Generation interrupted by user.")

        # Clean up the generator
        del generator

        if self.timings:
            prompt_time = first_token_timestamp - started_timestamp
            run_time = time.time() - first_token_timestamp
            print(f"\nPrompt length: {len(input_tokens)}, New tokens: {len(new_tokens)}")
            print(f"Time to first token: {prompt_time:.2f}s, Prompt tokens per second: {len(input_tokens) / prompt_time:.2f} tps")
            print(f"New tokens per second: {len(new_tokens) / run_time:.2f} tps")

        # Return the full response
        return ''.join(response)

# Example usage:
# handler = PhiONNXModelHandler(verbose=True, timings=True)
# handler.load_model("path_to_onnx_model")
# args = {
#     'max_length': 100,
#     'temperature': 0.7,
#     'top_k': 50
# }
# handler.chat_mode(args)
