import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from dashboard.reader import (
    get_configuration,
    get_evaluation_metrics,
    get_experiment,
    get_experiments,
    get_training_metrics,
)

load_dotenv()

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)


def display_experiment_details(experiment):
    st.subheader(f"Experiment: {experiment.label}")
    st.write(f"**Started at:** {experiment.created_at}")
    if experiment.ended_at:
        st.write(f"**Ended at:** {experiment.ended_at}")
        duration = experiment.ended_at - experiment.created_at
        st.write(f"**Duration:** {duration}")
    else:
        st.write("**Ended at:** Ongoing")


def display_configuration(config):
    if config:
        st.markdown("### Configuration")
        config_df = pd.DataFrame(
            {
                "Parameter": [
                    "Batch Size",
                    "Hidden Size",
                    "Learning Rate",
                    "Max Epochs",
                    "Data Split Ratio",
                    "Max Samples",
                ],
                "Value": [
                    config.batch_size,
                    config.hidden_size,
                    config.learning_rate,
                    config.max_epochs,
                    config.data_split_ratio,
                    config.max_samples,
                ],
            }
        )
        st.table(config_df.set_index("Parameter"))


def display_training_metrics(training_metrics):
    if training_metrics:
        st.markdown("### Training Metrics")
        df_training = pd.DataFrame(
            [
                {
                    "Epoch": tm.epoch,
                    "Training Loss": tm.training_loss,
                    "Training Accuracy": tm.training_accuracy,
                    "Validation Loss": tm.validation_loss,
                    "Validation Accuracy": tm.validation_accuracy,
                }
                for tm in training_metrics
            ]
        )

        # Plot Loss vs. Epoch
        fig, ax = plt.subplots()
        ax.plot(
            df_training["Epoch"],
            df_training["Training Loss"],
            label="Training Loss",
            marker="o",
        )
        ax.plot(
            df_training["Epoch"],
            df_training["Validation Loss"],
            label="Validation Loss",
            marker="o",
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss vs Epoch")
        ax.legend()
        st.pyplot(fig)

        # Plot Accuracy vs. Epoch
        fig, ax = plt.subplots()
        ax.plot(
            df_training["Epoch"],
            df_training["Training Accuracy"],
            label="Training Accuracy",
            marker="o",
        )
        ax.plot(
            df_training["Epoch"],
            df_training["Validation Accuracy"],
            label="Validation Accuracy",
            marker="o",
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy vs Epoch")
        ax.legend()
        st.pyplot(fig)

    else:
        st.write("No training metrics available.")


def display_evaluation_metrics(evaluation_metrics):
    if evaluation_metrics:
        st.markdown("### Evaluation Metrics")
        for eval_metrics in evaluation_metrics:
            st.markdown(f"#### Test Data Path: {eval_metrics.test_data_path}")
            eval_df = pd.DataFrame(
                {
                    "Metric": ["Test Loss", "Test Accuracy"],
                    "Value": [eval_metrics.test_loss, eval_metrics.test_accuracy],
                }
            )
            st.table(eval_df.set_index("Metric"))
    else:
        st.write("No evaluation metrics available.")


def display_checkpoint_metadata(checkpoint_metadata):
    if checkpoint_metadata:
        st.markdown("### Checkpoint Metadata")
        checkpoint_df = pd.DataFrame(
            [
                {
                    "Epoch": cm.epoch,
                    "Model Path": cm.model_path,
                    "File Size (B)": cm.file_size,
                    "Hidden Size": cm.hidden_size,
                }
                for cm in checkpoint_metadata
            ]
        )
        st.table(checkpoint_df.set_index("Epoch"))
    else:
        st.write("No checkpoint metadata available.")


def display_training_dataset_metadata(dataset_metadata):
    if dataset_metadata:
        st.markdown("### Training Dataset Metadata")
        dataset_df = pd.DataFrame(
            [
                {
                    "Dataset Path": dataset_metadata.dataset_path,
                    "Checksum": dataset_metadata.checksum,
                }
            ]
        )
        st.table(dataset_df.set_index("Dataset Path"))
    else:
        st.write("No dataset metadata available.")


def display_experiment_options(experiments):
    experiment_options = {exp.id: exp.label for exp in experiments}
    selected_experiment_id = st.selectbox(
        "Select an Experiment",
        options=experiment_options.keys(),
        format_func=lambda x: experiment_options[x],
    )
    return selected_experiment_id


def display_dashboard():
    st.title("Experiment Dashboard")
    session = Session()

    experiments = get_experiments(session)
    selected_experiment_id = display_experiment_options(experiments)

    selected_experiment = get_experiment(session, selected_experiment_id)

    if selected_experiment:
        display_experiment_details(selected_experiment)
        display_configuration(get_configuration(session, selected_experiment_id))
        display_training_metrics(get_training_metrics(session, selected_experiment_id))
        display_evaluation_metrics(
            get_evaluation_metrics(session, selected_experiment_id)
        )
        display_checkpoint_metadata(selected_experiment.checkpoint_metadata)
        display_training_dataset_metadata(selected_experiment.training_dataset_metadata)

    session.close()
