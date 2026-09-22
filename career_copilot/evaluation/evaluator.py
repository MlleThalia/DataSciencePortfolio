
import logging

def f1_score(precision, recall)->float:
    """Returns a F1-score"""
    if precision+recall == 0 :
        return 0
    return 2 * ((precision*recall) / (precision+recall))

def precision(true_positives, false_positives)->float:
    """Returns a Precision""" 
    if true_positives+false_positives == 0 :
        return 0
    return true_positives / (true_positives+false_positives)

def recall(true_positives, false_negatives)->float: 
    """Returns a Recall"""
    if true_positives+false_negatives == 0 :
        return 0
    return true_positives / (true_positives+false_negatives)

def compute_evaluation_metrics(expected_list : list, predicted_list: list)->dict:
    """Returns F1-score, Precision and Recall"""
    predicted_set = set(predicted_list)
    expected_set = set(expected_list)

    true_positives = sum(1 for item in predicted_set if item in expected_set)
    false_positives = len(predicted_set) - true_positives
    false_negatives = len(expected_set) - true_positives

    precision_value = precision(true_positives, false_positives)
    recall_value = recall(true_positives, false_negatives)
    f1_score_value = f1_score(precision_value, recall_value)

    return {"precision" : precision_value,
            "recall" : recall_value,
            "f1_score" : f1_score_value
        }

from career_copilot.models.job_analyser import JobAnalysis

def evaluate_job_analysis(expected_job_analysis : JobAnalysis, predicted_job_analysis : JobAnalysis)->dict:
    """Evaluate a job analysis"""
    return {
        "skills" : compute_evaluation_metrics(expected_job_analysis.skills, predicted_job_analysis.skills),
        "technologies" : compute_evaluation_metrics(expected_job_analysis.technologies, predicted_job_analysis.technologies),
        "languages" : compute_evaluation_metrics(expected_job_analysis.languages, predicted_job_analysis.languages),
        "keywords" : compute_evaluation_metrics(expected_job_analysis.keywords, predicted_job_analysis.keywords),
    }