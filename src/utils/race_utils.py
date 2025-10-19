def is_city_circuit(race_name):
    names = ['Monaco', 'Singapore', 'Azerbaijan', 'Saudi', 'Australian', 'Vegas', 'Miami', 'Canadian']

    for name in names:
        if name in race_name:
            return True
    return False

def is_night_race(race_name):
    names = ['Bahrain', 'Saudi Arabian', 'Abu Dhabi', 'Singapore', 'Qatar', 'Las Vegas', 'Sakhir']
    for name in names:
        if name in race_name:
            return True
    return False