from tracking.models import (
    Configuration,
    EvaluationMetrics,
    Experiment,
    TrainingMetrics,
)


def get_experiments(session):
    return session.query(Experiment).all()


def get_experiment(session, experiment_id):
    return session.query(Experiment).filter_by(id=experiment_id).first()


def get_configuration(session, experiment_id):
    return session.query(Configuration).filter_by(experiment_id=experiment_id).first()


def get_training_metrics(session, experiment_id):
    return session.query(TrainingMetrics).filter_by(experiment_id=experiment_id).all()


def get_evaluation_metrics(session, experiment_id):
    return session.query(EvaluationMetrics).filter_by(experiment_id=experiment_id).all()
