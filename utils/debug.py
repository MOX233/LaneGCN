
def generate_FL_table_with_only_one_car(FL_table):
    FL_table_with_only_one_car = {}
    for k,v in FL_table.items():
        FL_table_with_only_one_car[k] = {'car_0':[]}
    return FL_table_with_only_one_car
