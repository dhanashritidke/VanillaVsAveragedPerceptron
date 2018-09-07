import io
def read(file_path,learn=True):
    output=[]
    with io.open(file_path,'r',encoding='utf-8') as file:
        lines = file.readlines()
        output=[]
        for line in lines:
            output.append({})
            words=line.split()[:3]
            id, trueFake, posNeg, review= words[0], words[1], words[2], line.split(' ', 3)[3]
            output[-1][id] = (trueFake, posNeg, review)
    return output