from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Experiment(Base):
    __tablename__ = "experiment"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))
    ended_at = Column(DateTime(timezone=True), default=None)
    training_data_id = Column(
        Integer, ForeignKey("dataset_metadata.id"), nullable=False
    )
    label = Column(String)

    configuration = relationship("Configuration")
    training_metrics = relationship("TrainingMetrics")
    evaluation_metrics = relationship("EvaluationMetrics")
    checkpoint_metadata = relationship("CheckpointMetadata")
    training_dataset_metadata = relationship("DatasetMetadata")


class Configuration(Base):
    __tablename__ = "configuration"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))
    batch_size = Column(Integer, nullable=False)
    hidden_size = Column(Integer, nullable=False)
    learning_rate = Column(Float, nullable=False)
    data_path = Column(String, nullable=False)
    data_split_ratio = Column(Float, nullable=False)
    max_epochs = Column(Integer, nullable=False)
    output_path = Column(String, nullable=False)
    max_samples = Column(Integer)
    experiment_id = Column(Integer, ForeignKey("experiment.id"), nullable=False)


class TrainingMetrics(Base):
    __tablename__ = "training_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))
    training_loss = Column(Float, nullable=False)
    training_accuracy = Column(Float, nullable=False)
    validation_loss = Column(Float, nullable=False)
    validation_accuracy = Column(Float, nullable=False)
    epoch = Column(Integer, nullable=False)
    experiment_id = Column(Integer, ForeignKey("experiment.id"), nullable=False)

    __table_args__ = (
        UniqueConstraint("epoch", "experiment_id", name="uq_epoch_experiment"),
    )


class EvaluationMetrics(Base):
    __tablename__ = "evaluation_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))
    test_loss = Column(Float, nullable=False)
    test_accuracy = Column(Float, nullable=False)
    test_data_path = Column(String, nullable=False)
    experiment_id = Column(Integer, ForeignKey("experiment.id"), nullable=False)


class CheckpointMetadata(Base):
    __tablename__ = "checkpoint_metadata"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))
    model_path = Column(String, nullable=False)
    experiment_id = Column(Integer, ForeignKey("experiment.id"), nullable=False)
    hidden_size = Column(Integer, nullable=False)
    file_size = Column(Integer, nullable=False)
    epoch = Column(Integer, nullable=False)


class DatasetMetadata(Base):
    __tablename__ = "dataset_metadata"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))
    dataset_path = Column(String, nullable=False)
    checksum = Column(String, nullable=True)
