# --*-- encoding: utf-8 --*--

import json
from random import randint


# parse json files, add similar sentences to evidence

# load json file and parse
def parseJsonFile(filename):
    f = open(filename)
    lines = f.readlines()
    f.close()
    if lines and len(lines) > 0:
        content = json.loads(''.join(lines))
        return content
    return None

def convertEvidence(entry, question, entryIndex):
    answer = entry.get('answer')
    # array to string
    if answer and len(answer) > 0:
        answer = answer[0]
    if answer == 'no_answer':
        answer = ''
    evidence = entry.get('evidence')
    evidenceSpace = ' '.join(w for w in evidence)

    # find occurrences in evidence
    answers = []
    answer_start = 0
    while len(answer) > 0 and answer_start < len(evidence):
        answer_start = evidence.find(answer, answer_start)
        if answer_start < 0:
            break
        answer_text = ' '.join(w for w in answer) 
        answers.append({
            "answer_start" : answer_start * 2,
            "text": answer_text
        })
        answer_start += len(answer)

    if len(answers) < 1:
        return None

    qas = [{
        "id": str(entryIndex) + 'qa' + str(randint(1, 1000000)),
        "question": question,
        "answers": answers
    }]

    return {
        "title": "None",
        "paragraphs": [{
            "context": evidenceSpace,
            "qas": qas
        }]
    }

# convert one entry at a time
def webqa2squad(trainEntry, entryIndex):
    question = trainEntry.get('question')
    evidences = trainEntry.get('evidences')
    if not question or not evidences:
        return None

    # {answer, evidence} ==> { title, paragraphs }
    result = []
    questionSpace = ' '.join(w for w in question)
    for k in evidences.keys():
        evid = evidences[k]
        article = convertEvidence(evid, questionSpace, entryIndex)
        if article:
            result.append(article)
    return result

def checkAnswerStart(item):
    context = item['paragraphs'][0]['context']
    answers = item['paragraphs'][0]['qas'][0]['answers']
    for ans in answers:
        start = ans['answer_start']
        text = ans['text']
        if text == context[start : start + len(text)]:
            print('OK: ', start, text)
        else:
            print('>>>>> Error: ', start, text, context[start : start + len(text)])


def simpleConvert(filename):
    content = parseJsonFile(filename)
    keys = list(content.keys())
    trainSet = []
    testSet  = []
    for i in range(len(keys)):
        entry = content[keys[i]]
        if randint(1, 100) <= 2:
            testSet.extend(webqa2squad(entry, i))
        else:
            trainSet.extend(webqa2squad(entry, i))
    return (trainSet, testSet)

def writeFile(dataset, filename):
    with open(filename, 'w') as fp:
        json.dump({
            "data": dataset
        }, fp)

def convertWebqa(infile, outdir):
    train, test = simpleConvert(infile)
    writeFile(train, outdir + '/train-v1.1.json')
    writeFile(test,  outdir + '/dev-v1.1.json')

# extract sentences from evidence for experiments
def getAllEvidences(trainEntry):
    evidences = trainEntry.get('evidences')
    if not evidences or len(evidences) < 1:
        return []
    return [evidences[k].get('evidence') for k in evidences.keys()]

def getAllChineseCharacters(jsonData):
    words = set()
    for article in jsonData['data']:
        for para in article['paragraphs']:
            context = para['context']
            words = words | set(context.split(' '))
            for qa in para['qas']:
                question = qa['question']
                words = words | set(question.split(' '))
                for answer in qa['answers']:
                    answer_text = answer['text']
                    words = words | set(answer_text.split(' '))
    return words

# simple version of edit distance
def levDistance(s1, len1, s2, len2):
    if len1 == 0:
        return len2
    if len2 == 0:
        return len1
    cost = 1
    if (s1[len1 - 1] == s2[len2 - 1]):
        cost = 0
    return min(
        levDistance(s1, len1 - 1, s2, len2) + 1,
        levDistance(s1, len1, s2, len2 - 1) + 1,
        levDistance(s1, len1 - 1, s2, len2 - 1) + cost
    )

# Jaccard Similarity Coefficient
def naiveSimilarity(s1, s2):
    set1 = set(s1)
    set2 = set(s2)
    return len(s1 & s2) / len(s1 | s2)


def simpleIDFOverlap(s1, s2):
    words = set(s1) | set(s2)
    return words

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        inJson = sys.argv[1]
        outDir = sys.argv[2]
        convertWebqa(inJson, outDir)
