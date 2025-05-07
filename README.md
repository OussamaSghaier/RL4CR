CoRAL: Leveraging Reward Models for Guiding Code Review Comment Generation
===============================
This is the replication package accompanying our paper, *Leveraging Reward Models for Guiding Code Review Comment Generation*.


Overview
---
We propose a novel framework, called CoRAL,
that leverages reinforcement learning to generate code review
comments. Our framework employs two reward strategies,
namely *semantic similarity* and *subsequent task* rewards, to
guide the generation process. In the comment generation task,
the LLM takes the patch (i.e., code difference) as input and
generates a comment. 

- For the subsequent task reward strategy,
the generated comment, along with the patch, is fed into
the LLM to produce the necessary code edits. The reward
value is determined by measuring the correctness of code
edits using metrics such as loss or crystalBLEU to compare
the generated code edits with real ones. This strategy aims to
generate useful comments that facilitate effective code refinement. 

- For the semantic similarity reward
strategy, the reward value is the semantic similarity
between the generated and real comments. This approach
ensures that the generated comments, while potentially phrased
differently, convey the same meaning as the real comments,
thus maintaining their relevance and usefulness.


Project structure
---
    .
    ├── code_refinement             # Code refinement package
        ├── SFT                     # Code for supervised fine-tuning on code refinement
    ├── comment_generation          # Comment generation package
        ├── Reward                  # Reward models package
            ├── coderef_nxt_task    # Subsequent task reward models package
            ├── semantic_similarity # Semantic similarity reward models package
        ├── RL                      # Reinforcement learning package
            ├── semantic_similarity # Code for reinforcement learning with semantic similarity reward
            ├── subsequent_task     # Code for reinforcement learning with subsequent task reward
                ├── crystal_bleu    # Code for reinforcement learning with subsequent task crystal bleu
                ├── loss            # Code for reinforcement learning with subsequent loss
        ├── SFT                     # Code for supervised fine-tuning on comment generation
    ├── data                        # Folder that contains the datasets
        ├── code_refinement         # Folder that contains the code refinement dataset
        ├── comment generation      # Folder that contains the comment generation dataset
    ├── data_viz                    # Folder that contains notebooks used to visualize and 
    ├── evaluation                  # Package with generic evaluation scripts (bleu, crystal bleu, etc.)
    ├── test                        # Package that has scripts used for test purposes
    ├── utils                       # Utilities package (e.g., configuration)
    ├── Dockerfile                  # Dockerfile to setup the environment


Setup
---
To facilitate the usage and results replication, CoRAL can be trained, run, and evaluated inside a Docker container. To build the Docker image and use the different scripts to identify microservices for a specific project.

1. Clone the CoRAL repository:
    ```bash
    git clone https://github.com/RL4CR/CoRAL.git
    ```

2. Navigate to the CoRAL directory:
    ```bash
    cd CoRAL
    ```

3. Build the Docker image:
    ```bash
    docker build -t coral-image .
    ```

4. Create a docker container from the built image and connect to the running Docker container:
    ```bash
    docker run -it --name coral-container coral-image
    ```

    The above command should be run once (the first time) to create a container from the built image. 
    For subsequent attempts, we should use the following commands to start and connect to the created docker container:
    ```bash
    docker start coral-container
    ```

    To stop the container:
    ```bash
    docker stop coral-container
    ```

    CoRAL is now set up to run within a Docker container for your project, and you can easily start, connect, and stop the container as needed.


Example usage
---

- ### Example 1

To fine-tune CodeLlama-7B on comment generation using supervised fine-tuning (SFT), please yse the following command:

```bash
python code_refinement/SFT/sft.py \
        --model_name=codellama/CodeLlama-7b-Instruct-hf
        --train_data=data/comment_generation/msg_traing.jsonl \
        --checkpoint_folder=data/ckpt_comments_sft \
        --num_epochs=5 \
        --max_sequence_length=2048 \
        --learning_rate=3e-4 \
        --gradient_accumulation_steps=4 \
        --batch_size=8 \
        --eval_steps=5000 \
        --save_steps=1000 \
        --log_steps=100
        --output_dir=data/output_comments_sft
```
The description of the different arguments is given in the configuration file in *code_refinement/SFT/config.py*.


- ### Example 2

To fine-tune CodeLlama-7B (or any other model) using reinforcement learning and semantic similarity as a reward model, you can use this command:

```bash
python code_refinement/RL/ppo.py \
        --reward_model=semantic \
        --model_name=codellama/CodeLlama-7b-Instruct-hf
        --train_data=data/comment_generation/msg_traing.jsonl \
        --checkpoint_folder=data/ckpt_comments_sft \
        --num_epochs=5 \
        --output_dir=data/output_comments_sft
```

Please note that parameters default values, that are defined in the configuration file, are used during the training if not specified explicitly as arguments.


Citing
---
TODO

Contact us
---
TODO

