import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from dashboard.dashboard import DATABASE_URL
from tracking.models import (
    CheckpointMetadata,
    Configuration,
    DatasetMetadata,
    EvaluationMetrics,
    Experiment,
    TrainingMetrics,
)

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(bind=engine)


def persist_model(model: Experiment | Configuration):
    """Persist a given model instance to the database."""
    session = SessionLocal()
    try:
        session.add(model)
        session.commit()
        session.refresh(model)
        return model
    except Exception as e:
        session.rollback()
        print(f"Error creating entry: {e}")
    finally:
        session.close()


def log_new_experiment(training_data_id, label=None):
    """Log a new experiment and return its ID."""
    experiment = Experiment(training_data_id=training_data_id, label=label)
    persisted_experiment = persist_model(experiment)

    return persisted_experiment.id


def log_experiment_ended(experiment_id):
    """Mark an experiment as ended by updating its end timestamp."""
    session = SessionLocal()
    try:
        experiment = session.query(Experiment).filter_by(id=experiment_id).first()
        experiment.ended_at = datetime.now(timezone.utc)
        session.commit()
    finally:
        session.close()


def log_configuration(experiment_id, config_data):
    """Log configuration details for an experiment."""
    new_config = Configuration(
        batch_size=config_data["batch_size"],
        hidden_size=config_data["hidden_size"],
        learning_rate=config_data["learning_rate"],
        data_path=str(config_data["data_path"]),
        data_split_ratio=config_data["data_split_ratio"],
        max_epochs=config_data["max_epochs"],
        output_path=str(config_data["output_path"]),
        max_samples=config_data["max_samples"],
        experiment_id=experiment_id,
    )

    persist_model(new_config)


def log_training_metrics(experiment_id, epoch, train_metrics, val_metrics):
    """Log training metrics including loss and accuracy for each epoch."""
    metric = TrainingMetrics(
        training_loss=train_metrics["loss"],
        training_accuracy=train_metrics["accuracy"],
        validation_loss=val_metrics["loss"],
        validation_accuracy=val_metrics["accuracy"],
        epoch=epoch,
        experiment_id=experiment_id,
    )

    persist_model(metric)


def log_evaluation_metrics(experiment_id, eval_metrics, data_path):
    """Log evaluation metrics including test loss and accuracy."""
    metric = EvaluationMetrics(
        test_loss=eval_metrics["loss"],
        test_accuracy=eval_metrics["accuracy"],
        test_data_path=str(data_path),
        experiment_id=experiment_id,
    )

    persist_model(metric)


def log_checkpoint(experiment_id, model_path, epoch, hidden_size):
    """Log checkpoint metadata including model path and file size."""
    file_size = os.path.getsize(model_path)

    metadata = CheckpointMetadata(
        model_path=str(model_path),
        experiment_id=experiment_id,
        epoch=epoch,
        file_size=file_size,
        hidden_size=hidden_size,
    )

    persist_model(metadata)


def log_dataset_metadata(dataset_path):
    """Log dataset metadata and return the metadata ID."""

    checksum = compute_file_checksum(dataset_path)

    if existing_metadata := find_dataset_metadata_by_checksum(checksum):
        return existing_metadata.id

    metadata = DatasetMetadata(dataset_path=str(dataset_path), checksum=checksum)

    persist_model(metadata)

    return metadata.id


def find_dataset_metadata_by_checksum(checksum):
    """Find dataset metadata based on file checksum."""
    session = SessionLocal()
    try:
        return session.query(DatasetMetadata).filter_by(checksum=checksum).first()
    finally:
        session.close()


def compute_file_checksum(path: Path) -> str:
    """Compute and return the MD5 checksum of a file."""
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()
