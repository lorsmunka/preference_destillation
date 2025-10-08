from time import time


class ModelHandler:
    def __init__(self):
        print("ModelHandler initialized")

    def generate_training_example(self, sentence):
        start_time = time()

        elapsed_time = time() - start_time
        print(
            f"Generated training example -> took {elapsed_time:.2f} seconds.")

        return {
            "sentence": sentence,
            "generated_text": f"Generated text for: {sentence}"
        }
