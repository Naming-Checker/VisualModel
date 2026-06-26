list:
    just --list

reevaluate-fine-tuned-model:
    python src/embedding.py --model 'models/similarity.safetensors' --target 'models/logos_embedding_tuned.pt'
    python src/similarity_eval.py --embeddings-path 'models/logos_embedding_tuned.pt'
