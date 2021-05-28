from tqdm import tqdm

from predict_single import predict


if __name__ == '__main__':
    with open('test_qa.txt', 'r') as file:
        content = file.readlines()

    context_questions = []
    for line in content:
        qq = {'context': line.split(' ,')[0], 'question': line.split(' ,')[1]}
        context_questions.append(qq)

    print("[] Starting predictions...")
    results = []
    for idx, qq in enumerate(tqdm(context_questions)):
        results.append(predict(context=qq['context'], question=qq['question'].rstrip(), id=idx, model='model_8'))

    print("[] Writing results...")
    with open('results.txt', 'a') as file:
        for ans in results:
            file.write(f"{ans}\n")
