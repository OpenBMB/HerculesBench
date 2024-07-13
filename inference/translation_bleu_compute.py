from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

def calculate_sentence_bleu(reference, candidate):
    """
    Calculate BLEU score for a single sentence.
    :param reference: Reference translation, a string.
    :param candidate: Candidate translation, a string.
    :return: BLEU score
    """
    # Remove punctuation and special characters
    reference = reference.lower()
    candidate = candidate.lower()
    # Tokenize
    reference_tokens = reference.split()
    candidate_tokens = candidate.split()

    if len(reference_tokens) > 1 or len(candidate_tokens) > 1:
        # Calculate BLEU score, using smoothing function to avoid zero denominator
        smoothie = SmoothingFunction().method1
        return sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothie)
    else:
        return reference == candidate


# Example usage
# reference_text = "apple"
# candidate_text = "an apple"
# bleu_score = calculate_sentence_bleu(reference_text, candidate_text)
# print(f"BLEU score: {bleu_score}")
