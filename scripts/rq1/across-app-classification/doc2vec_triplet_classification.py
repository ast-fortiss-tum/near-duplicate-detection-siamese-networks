import torch
import torch.optim as optim
import os
import time
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[3]))

from scripts.rq1.datasets import prepare_datasets_and_loaders_across_app_triplet
from scripts.utils.embedding import run_embedding_pipeline_doc2vec
from scripts.rq1.test import test_model_triplet
from scripts.rq1.train import train_one_epoch_triplet
from scripts.rq1.validate import validate_model_triplet
from scripts.utils.networks import TripletSiameseNN
from scripts.utils.utils import (
    set_all_seeds,
    initialize_weights,
    save_results_to_excel,
    load_pairs_from_db,
    initialize_device, create_folders_if_not_exist
)

##############################################################################
#      Main Functions  Doc2Vec Triplet AcrossApp Classification             #
##############################################################################

if __name__ == "__main__":
    seed = 42
    set_all_seeds(seed)
    device = initialize_device()
    base_path    = os.getcwd()

    selected_apps = [
        'addressbook', 'claroline', 'ppma', 'mrbs',
        'mantisbt', 'dimeshift', 'pagekit', 'phoenix','petclinic'
    ]

    table_name    = "nearduplicates"
    db_path       = f"{base_path}/dataset/SS_refined.db"
    dom_root_dir  = f"{base_path}/resources/doms"
    results_dir    = f"{base_path}/results/rq1"
    model_dir     = f"{base_path}/models"
    emb_dir       = f"{base_path}/embeddings"
    title         = "acrossapp_doc2vec"
    setting_key   = "triplet"
    save_results  = True

    doc2vec_path    = f"{base_path}/resources/embedding-models/content_tags_model_train_setsize300epoch50.doc2vec.model"

    chunk_size    = 512
    batch_size    = 128
    num_epochs    = 7
    lr            = 1e-4
    weight_decay  = 0.01
    chunk_limit   = 2
    overlap       = 0
    margin        = 1

    results = []
    create_folders_if_not_exist([model_dir, emb_dir, results_dir])

    for test_app in selected_apps:

        print("\n=============================================")
        print(f"[Info] Starting across-app iteration: test_app = {test_app}")
        print("=============================================")
        preprocess_start_time = time.time()

        model_filename = f"{title}_{setting_key}_{test_app}_cl_{chunk_limit}_bs_{batch_size}_ep_{num_epochs}_lr_{lr}_wd_{weight_decay}.pt"
        model_file = os.path.join(model_dir, model_filename)

        all_pairs = load_pairs_from_db(db_path, table_name, selected_apps)
        if not all_pairs:
            print("[Warning] No data found in DB with is_retained=1. Skipping.")
            continue
        print(f"[Info] Total pairs: {len(all_pairs)}")

        state_embeddings, final_input_dim = run_embedding_pipeline_doc2vec(
            pairs_data=all_pairs,
            dom_root_dir=dom_root_dir,
            doc2vec_model_path=doc2vec_path,
            cache_path=os.path.join(emb_dir, f"{title}_cache_{test_app}.pkl")
        )
        if not state_embeddings or (final_input_dim == 0):
            print("[Warning] No embeddings found. Skipping.")
            continue

        train_loader, val_loader, test_loader = prepare_datasets_and_loaders_across_app_triplet(
            all_pairs,
            test_app=test_app,
            state_embeddings=state_embeddings,
            batch_size=batch_size,
            seed=seed
        )

        model = TripletSiameseNN(input_dim=final_input_dim)
        initialize_weights(model, seed)
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        preprocess_time = time.time() - preprocess_start_time

        if os.path.exists(model_file):
            print(f"[Info] Found saved model for {test_app} at {model_file}. Loading model and skipping training.")
            model.load_state_dict(torch.load(model_file, weights_only=True))
            training_time = "N/A"
        else:
            print(f"[Info] No saved model for {test_app}. Training will start.")
            start_time = time.time()

            for epoch in range(num_epochs):
                train_loss = train_one_epoch_triplet(model, train_loader, optimizer, device, epoch, num_epochs, margin=margin)
                val_loss = validate_model_triplet(model, val_loader, device, threshold=0.5)
                print(f"  Epoch {epoch + 1}/{num_epochs} => Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            training_time = time.time() - start_time
            torch.save(model.state_dict(), model_file)
            print(f"[Info] Model saved to {model_file}")

        metrics_dict = test_model_triplet(model, test_loader, device, threshold=0.5)
        print(f"[Test Results] for test_app={test_app}: {metrics_dict}")

        row = {
            "TestApp": test_app,
            "Accuracy": metrics_dict["Accuracy"],
            "Precision": metrics_dict["Precision"],
            "Recall": metrics_dict["Recall"],
            "F1 Score (Weighted Avg)": metrics_dict["F1 Score (Weighted Avg)"],
            "F1_Class 0": metrics_dict["F1_Class 0"],
            "F1_Class 1": metrics_dict["F1_Class 1"],
            "TrainingTime": training_time,
            "PreprocessingTime": preprocess_time,
        }
        results.append(row)

    if save_results:
        save_results_to_excel(
            title=title,
            results=results,
            results_dir=results_dir,
            setting_key=setting_key,
            overlap=overlap,
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr,
            weight_decay=weight_decay,
            chunk_limit=chunk_limit
        )
