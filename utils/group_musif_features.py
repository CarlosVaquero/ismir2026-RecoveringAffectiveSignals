import re


module_regex_map = {
    ## Music21
    r"m21_.*": "music21",
    ## Core
    r"Measures": "core",
    r".+_SoundingMeasures": "core",
    r".+_SoundingMeasuresMean": "core",
    r".+_no.es": "core",
    r".+no.esMean": "core",
    r"NumberOfBeats": "core",
    ## Harmony (+ key + scale)
    r"Key": "harmony",
    r"KeySignature": "harmony",
    r"KeySignatureType": "harmony",
    r"Mode": "harmony",
    r"Key_.+": "harmony",
    r"KeySignature_.+": "harmony",    # melody (+ ambitus)
    r"KeySignatureType_.+": "harmony",  # e.g. KeySignatureType_s
    r"Mode_.+": "harmony",  
    r"Harmony_Analysis_Available": "harmony",
    r"Harmony_Chord_.+_Count": "harmony",
    r"Harmony_Chord_.+_Per": "harmony",
    r"Harmony_Chord_types_.+": "harmony",
    r"Harmony_Chords_Grouping_1.+_Count": "harmony",
    r"Harmony_Chords_Grouping_1.+_Per": "harmony",
    r"Harmony_Chords_Grouping_2.+_Count": "harmony",
    r"Harmony_Chords_Grouping_2.+_Per": "harmony",
    r"Harmony_HarmonicRhythm": "harmony",
    r"Harmony_HarmonicRhythmBeats": "harmony",
    r"Harmony_Key_.+_PercentageMeasures": "harmony",
    r"Harmony_Key_Modulatory_(b|#)?\\d": "harmony",
    r"Harmony_Key_Modulatory_.+": "harmony",
    r"Harmony_Numerals_.+_Count": "harmony",
    r"Harmony_Numerals_.+_Per": "harmony",
    r"Harmony_Additions_.+": "harmony",
    r".+Degree(b+|#|x)?\\d_Count": "harmony",
    r".+Degree(b+|#|x)?\\d_Per": "harmony",
    r".+_Degree.+_Count": "harmony",
    r".+_Degree.+_Per": "harmony",
    r".+_Degree.+_Count_relative": "harmony",
    r".+_Degree.+_Per_relative": "harmony",
    # Melody (+ intervals)
    r".+_Asc_avg_step_.+_win_.+": "melody",
    r".+_Acc_avg_abs_step_.+_win_.+": "melody",
    r".+_Asc_prp_step_.+_win_.+": "melody",
    r".+_Dsc_avg_step_.+_win_.+": "melody",
    r".+_Dsc_prp_step_.+_win_.+": "melody",
    r".+_Spe_avg_abs_step_.+_win_.+": "melody",
    r".+Score_Interval.+": "melody",
    r".+_Ambitus": "melody",
    r".+_HighestNoteIndex": "melody",
    r".+_LowestNoteIndex": "melody",
    r".+_HighestNote": "melody",
    r".+_LowestNote": "melody",
    r".+AbsoluteIntervallic.+": "melody",
    r".+AscendingIntervallic.+": "melody",
    r".*Score_Interval.+": "melody",
    r".*Interval.+": "melody",
    r".*StepwiseMotion.+": "melody",
    r".*_NotesMean": "melody",           # e.g. SoundVa_NotesMean
    r".*RepeatedNotes.+": "melody",      # e.g. SoundVa_RepeatedNotes_Per
    r".*Semitones.+": "melody",          # e.g. SoundVa_SmallestAbsoluteSemitonesAsc/Desc
    r".*StepwiseMotion.+": "melody", 
    r".*Leaps.+": "melody",
    r".*_Notes$": "melody", 
    # Rhythm
    r".+_AverageDuration": "rhythm",
    r".+_DottedRhythm": "rhythm",
    r".+_DoubleDottedRhythm": "rhythm",
    r".+_RhythmInt": "rhythm",
    r"TimeSignature": "rhythm",
    r"TimeSignatureGrouped": "rhythm",
    r"TimeSignature_.+": "rhythm",  # e.g. TimeSignature_3/8
    r"TempoGrouped\d_.+": "rhythm",       # e.g. TempoGrouped2_Fast, TempoGrouped2_Slow
    r"TimeSignatureGrouped_.+": "rhythm",
    r"Tempo_.+": "rhythm",
    r"NumericTempo": "rhythm",
    r"Tempo": "rhythm",
    r"TempoGrouped1": "rhythm",
    r"TempoGrouped2": "rhythm",
    # Dynamics
    r".+_DynAbruptness": "dynamics",
    r".+_DynGrad": "dynamics",
    r".+_DynMean": "dynamics",
    r".+_DynMean_weighted": "dynamics",
    # Texture (+ lyrics)
    r"Score_VoiceReg": "texture",
    r".+_Syllables": "texture",
    r".+_SyllabicRatio": "texture",
    r".+_VoicePresence": "texture",
    r"Presence_of_.+": "texture",
    r"Part.+|Part.+_Texture": "texture",
    r"Clef\d+_.+": "texture",            # e.g. Clef2_G-2
    r"Clef\d+": "texture",
    r"FamilyInstrumentation_.+": "texture", # e.g. FamilyInstrumentation_str
    r"FamilyScoring_.+": "texture", 
    r"FamilyInstrumentation": "texture",
    r"SoundScoring": "texture",
    r"Instrumentation": "texture",
    r"Scoring": "texture",
    r".+_NumberOfParts": "texture",
    r".+_NumberOfFilteredParts": "texture",
    r"Voices": "texture",
    r".+_Density": "texture",
    r".+_SoundingDensity": "texture",
    r"Instrumentation_.+": "texture",
    r"SoundScoring_.+": "texture",
    r"Scoring_.+": "texture", 
}

def get_module(feature_name: str) -> str:
    for pattern, module in module_regex_map.items():
        if re.fullmatch(pattern, feature_name):
            return module
    return "unknown"


def group_columns(columns):
    grouped = {}
    for col in columns:
        module = get_module(col)
        grouped.setdefault(module, []).append(col)
    return grouped