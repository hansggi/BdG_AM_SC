import dask

@dask.delayed
def increment_data(data):
    return data + 1

def do_analysis(result):
    return sum(result)

all_data = [1, 2, 3, 4, 5]
results = []
for data in all_data:
    data_incremented = increment_data(data)
    results.append(data_incremented)

# print(results)

analysis = dask.delayed(do_analysis)(results) # dask.delayed can also be used in 
                                              # in this manner 
print(analysis)
final_result = analysis.compute() # execute all delayed functions
print(f"and the final result is: {final_result}")