import numpy as np
from tensorflow.python.framework.errors import InvalidArgumentError


def get_result(student_func):
    """
    Run unit tests against <student_func>
    """
    answer = np.array([0.65900117, 0.24243298, 0.09856589])
    result = {
        'correct': False,
        'feedback': f'That\'s the wrong answer.  It should print {answer}',
        'comment': ''}

    try:
        output = student_func()
        if not isinstance(output, np.ndarray):
            result['feedback'] = 'Output is the wrong type.'
            result['comment'] = 'The output should come from running the session.'
        elif np.allclose(output, [2, 1, 0.1]):
            result['feedback'] = 'You\'re returning the logits.'
            result['comment'] = 'You need to apply softmax to the logits and return them.'
        elif np.allclose(output, answer):
            result['correct'] = True
            result['feedback'] = 'That\'s the correct softmax!'
    except InvalidArgumentError as err:
        if err.message.startswith('You must feed a value for placeholder tensor'):
            result['feedback'] = 'The placeholder is not being set.'
            result['comment'] = 'Try using the feed_dict.'
        else:
            raise

    return result

def run_grader(student_func):
    
    try:
    # Get grade result information
        result = get_result(student_func)
    except Exception as err:
        # Default error result
        result = {
            'correct': False,
            'feedback': 'Something went wrong with your submission:',
            'comment': str(err)}

    feedback = result.get('feedback')
    comment = result.get('comment')

    print(f"{feedback}\n{comment}\n")



if __name__ == "__main__":
    run_grader(student_func)
