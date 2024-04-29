

## Dependencies

To run this project, it is recommended to use Python 3.10.12 or higher, and have the following packages installed:

- **torch**: `2.2.1`
- **huggingface_hub**: `0.20.3`
- **transformers**: `4.40.0`

## Getting Started

```bash
cd ura_practice
```
### for task1.1
####
* for gpt2 :
    ```bash
    python prediction.py --input_text "Ludwig the cat" --model_name "gpt2" --top_k 10
    ```
* for gpt2-large :
    ```bash
    python prediction.py --input_text "Ludwig the cat" --model_name "gpt2-large" --top_k 10
    ```

### for task1.2
####
* for gpt2 :
    ```bash
    python finetuning.py  --model_name "gpt2" --learning_rate 5e-5 --input_text "Ludwig the cat" --target_text " stretches" --epochs 10 --save_path "./model_ft"
    ```

    for accumulate_steps_version:
    ```bash
    python finetuning_accumulate_steps.py  --model_name "gpt2" --learning_rate 5e-5 --input_text "Ludwig the cat" --target_text " stretches" --epochs 10 --save_path "./model_ft"
    ```
* for gpt2-large :
    ```bash
    python finetuning.py  --model_name "gpt2-large" --learning_rate 5e-5 --input_text "Ludwig the cat" --target_text " stretches" --epochs 10 --save_path "./model_ft"
    ```
### for task1.3
####
* for gpt2 :
    ```bash
    python prompt.py  --model_name "gpt2" --learning_rate 5e-2 --input_text "Ludwig the cat" --target_text " stretches" --epochs 20 --save_path "./model_ftp"
    ```
* for gpt2-large :
    ```bash
    python prompt.py  --model_name "gpt2-large" --learning_rate 1e-2 --input_text "Ludwig the cat" --target_text " stretches" --epochs 50 --save_path "./model_ftp"
    ```
