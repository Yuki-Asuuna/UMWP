# Data Sample:
# JSON Format
# {
#             "question_id": 1,
#             "question": "While on vacation, Megan took 15 pictures at the zoo and 18 at the museum. If she later deleted 31 of the pictures, how many pictures from her vacation did she still have?",
#             "answer": [
#                 "2.0"
#             ],
#             "answerable": true,
# 			  "relevant_ids": [
# 					2,
# 			   ],
# 			  "category": 0,
#             "source": "MAWPS_test"
# }
import json
import random
import jsonlines


class StandardDatasetExample():
    def __init__(self, id=None, question=None, answer=None, answerable=None, category=None, relevant_ids=None,
                 source=None):
        self.id = id
        self.question = question
        self.answer = answer
        self.answerable = answerable
        self.category = category
        self.relevant_ids = relevant_ids
        self.source = source


class StandardDataset():
    def __init__(self, d=None):
        if d is None:
            d = []
        self.data = d

    @classmethod
    def FromJSONLFile(cls, data_dir):
        data = []
        with jsonlines.open(data_dir, "r") as f:
            for item in f:
                data.append(
                    StandardDatasetExample(item["id"], item["question"], item["answer"], item["answerable"],
                                           item["category"], item["relevant_ids"], item["source"]))
        return cls(data)

    def AddExample(self, StandardDatasetExample):
        self.data.append(StandardDatasetExample)

    def Size(self) -> int:
        return len(self.data)

    def GetRandomSamples(self, n):
        random_samples = random.sample(self.data, n)
        new_dataset = StandardDataset()

        for sample in random_samples:
            new_dataset.AddExample(sample)

        return new_dataset

    def Merge(self, StandardDataset):
        self.data.extend(StandardDataset.data)
        # reassign id
        id = 1
        for item in self.data:
            item.id = id
            id += 1

    def FetchAll(self):
        return self.data

    def Write2File(self, file_name):
        with open(file_name, 'w') as f:
            for item in self.data:
                json.dump(item.__dict__, f)
                f.write('\n')
