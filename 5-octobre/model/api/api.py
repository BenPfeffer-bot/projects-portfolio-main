from mistralai import Mistral
import sys
from pathlib import Path
import os

sys.path.append("5-octobre")
from src.config import load_config, load_logger

config = load_config()
logger = load_logger()


class MistralClient:
    def __init__(self):
        self.mistral = self.get_mistral_client()
        if not self.mistral:
            raise RuntimeError("Failed to initialize Mistral client")
        self.created_jobs = None
        self.training_data = None
        self.retrieved_job = None
        self.canceled_jobs = None
        self.chat_response = None

    def get_mistral_client(self):
        """
        Create a Mistral client.
        Returns:
            Mistral: Initialized Mistral client or None if creation fails
        """
        try:
            logger.debug("Creating Mistral client")
            mistral = Mistral(api_key=config["API_KEY"]["MISTRAL_API_KEY"])
            logger.debug("Mistral client created successfully")
            return mistral
        except Exception as e:
            logger.error(f"Error getting Mistral client: {e}")
            return None

    def load_training_data(self, file_path="training_data.jsonl"):
        """
        Load training data from the database.
        Args:
            file_path (str): Path to the training data file
        Returns:
            str: Training file ID or None if loading fails
        """
        try:
            logger.debug("Loading training data")
            if not Path(file_path).exists():
                raise FileNotFoundError(f"Training data file not found: {file_path}")

            training_data = self.mistral.files.upload_file(
                file_path=file_path,
                file_name=os.path.basename(file_path),
                purpose="fine-tuning",
            )
            logger.debug(f"Training data loaded successfully: {training_data}")
            self.training_data = training_data
            return training_data
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            return None

    def create_fine_tuning_job(self):
        """
        Create a fine-tuning job.
        Returns:
            object: Created fine-tuning job or None if creation fails
        """
        try:
            logger.debug("Creating fine-tuning job")
            training_file_id = self.load_training_data()
            if not training_file_id:
                raise ValueError("Failed to load training data")

            self.created_jobs = self.mistral.fine_tuning.create_job(
                training_file_id=training_file_id,
                model_name="mistral-7b-instruct",
                hyperparameters={
                    "learning_rate": 0.0001,
                    "training_steps": 1000,
                },
                auto_start=True,
            )
            logger.debug(f"Fine-tuning job created successfully: {self.created_jobs}")
            return self.created_jobs
        except Exception as e:
            logger.error(f"Error creating fine-tuning job: {e}")
            return None

    def run_fine_tuning(self):
        """
        Run a fine-tuning job and get chat completion.
        Returns:
            object: Chat completion response
        """
        try:
            retrieved_job = self.retrieve_job()
            if not retrieved_job or not retrieved_job.fine_tuned_model:
                raise ValueError("No fine-tuned model available")

            self.chat_response = self.mistral.chat.complete(
                model=retrieved_job.fine_tuned_model,
                messages=[
                    {
                        "role": "user",
                        "content": "What is the best French restaurant in Paris?",
                    },
                ],
            )
            return self.chat_response
        except Exception as e:
            logger.error(f"Error running fine-tuning: {e}")
            return None


if __name__ == "__main__":
    mistral_client = MistralClient()
    mistral_client.create_fine_tuning_job()
