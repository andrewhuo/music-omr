import sys
from music21 import converter

def extract_unique_measures(musicxml_path: str):
    """
    Parse a MusicXML/MXL file and return a sorted list of unique measure numbers.
    """
    score = converter.parse(musicxml_path)

    measure_set = set()
    for part in score.parts:
        for measure in part.getElementsByClass('Measure'):
            num = measure.measureNumber
            try:
                measure_num = int(num)
            except Exception:
                continue
            measure_set.add(measure_num)

    sorted_measures = sorted(measure_set)
    return sorted_measures

if __name__ == "__main__":
    path = sys.argv[1]
    measures = extract_unique_measures(path)
    for m in measures:
        print(m)
