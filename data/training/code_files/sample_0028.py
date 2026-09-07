def is_anagram(s1, s2):
    return sorted(s1.lower().replace(' ', '')) == sorted(s2.lower().replace(' ', ''))
