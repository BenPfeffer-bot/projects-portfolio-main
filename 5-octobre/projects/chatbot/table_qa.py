# src/table_qa.py
import pandas as pd
from transformers import TapasTokenizer, TapasForQuestionAnswering


class TableQuestionAnswering:
    def __init__(self, model_name="google/tapas-base-finetuned-wtq"):
        self.tokenizer = TapasTokenizer.from_pretrained(model_name)
        self.model = TapasForQuestionAnswering.from_pretrained(model_name)

    def answer_question(self, table: pd.DataFrame, question: str) -> str:
        """
        Answer a question given a Pandas DataFrame table using the TAPAS model.
        """
        inputs = self.tokenizer(table=table, queries=[question], padding="max_length", return_tensors="pt")
        outputs = self.model(**inputs)

        # Convert logits to predictions
        predicted_answer_coordinates, predicted_aggregation_indices = self.tokenizer.convert_logits_to_predictions(inputs, outputs.logits, outputs.logits_agg)

        # If no cells are selected, return "No answer."
        if not predicted_answer_coordinates or len(predicted_answer_coordinates[0]) == 0:
            return "No answer found."

        answer_cells = []
        for coord in predicted_answer_coordinates[0]:
            answer_cells.append(str(table.iat[coord]))

        # Check aggregation
        predicted_aggregation = predicted_aggregation_indices[0]
        # Aggregation: 0 = NONE, 1 = SUM, 2 = AVERAGE, 3 = COUNT, ...
        if predicted_aggregation == 0:
            # Just return the concatenation of selected cells
            final_answer = " ".join(answer_cells)
        else:
            # Perform aggregation if needed
            numeric_values = []
            for val in answer_cells:
                # Attempt to convert to float
                try:
                    numeric_values.append(float(val))
                except ValueError:
                    # Not numeric, just return cell values as-is
                    numeric_values = []
                    break

            if predicted_aggregation == 1 and numeric_values:
                final_answer = str(sum(numeric_values))
            elif predicted_aggregation == 2 and numeric_values:
                final_answer = str(sum(numeric_values) / len(numeric_values))
            elif predicted_aggregation == 3:
                final_answer = str(len(answer_cells))
            else:
                final_answer = " ".join(answer_cells)

        return final_answer.strip()
