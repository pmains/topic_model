"""
Class that takes a function which returns a generator
When called, it returns a generator that yields the same values as the function
When the generator is exhausted, it raises the StopIteration exception, then
    the function is called again to create a new generator
"""


class Reiterable:
    def __init__(self, generator_function, *args, **kwargs):
        self.generator_function = generator_function
        self.generator = None
        self.iteration = 0
        self.name = generator_function.__name__

        # store the arguments to the generator function
        self.args = args if args is not None else []
        print("args: ", self.args)
        self.kwargs = kwargs if kwargs is not None else {}
        print("kwargs: ", self.kwargs)

    def __iter__(self):
        return self

    def __next__(self):
        if self.generator is None:
            self.iteration += 1
            print(f"Creating new {self.name} generator {self.iteration}...")
            self.generator = self.generator_function(*self.args, **self.kwargs)
        try:
            return next(self.generator)
        except StopIteration as stop_iteration:
            self.generator = None
            raise stop_iteration


if __name__ == "__main__":
    # functions
    def iter10():
        for val in range(10):
            yield val


    rg = Reiterable(iter10)

    for i in rg:
        print("Iter 1: ", i)

    for j in rg:
        print("Iter 2: ", j)
