from clingo.control import Control

class Clingo:
    def __call__(self, program):
        program_str = str(program)
        ctl = Control()
        ctl.add("base", [], program_str)
        ctl.ground([("base", [])])

        models = []
        with ctl.solve(yield_=True) as hnd:
            for m in hnd:
                models.append({
                    "number": m.number,
                    "cost": m.cost,
                    "atoms": m.symbols(atoms=True)
                })
        del ctl
        return models

    def classify(self, program, class_name = "class", num_class_args=1):
        models = self(program + "\n#show {}/{}.".format(class_name, num_class_args))
        atoms = [str(atom) for atom in models[0]["atoms"] if atom.name == class_name and len(atom.arguments) == num_class_args]
        predictions = [")".join("(".join(predict.split('(')[1:]).split(')')[:-1]) for predict in atoms]
        predicted_labels = [(prediction.split(',')[0], prediction.split(',')[1], ",".join(prediction.split(',')[2:])) for prediction in predictions]
        return predicted_labels

    def optimise(self, program, class_names = ["class"], num_class_args=[1]):
        show_lines = "\n".join(["#show {}/{}.".format(class_name, num_args) for class_name, num_args in zip(class_names, num_class_args)])
        models = self(program + show_lines)
        answer_set = min(models, key=lambda model: model["cost"])["atoms"]
        answer_set = [str(atom) for atom in answer_set if atom.name in class_names and len(atom.arguments) in num_class_args]
        results = {
            class_name: []
            for class_name in class_names
        }
        for prediction in answer_set:
            predicate_name = prediction.split('(')[0]
            predicate_terms = ")".join("(".join(prediction.split('(')[1:]).split(')')[:-1])
            results[predicate_name].append(self._split_by_uppermost_commas(predicate_terms))
        return results

    def _split_by_uppermost_commas(self, input_str):
        open_counter = 0
        splits = []
        curr_split = ""
        for c in input_str:
            if c==',' and open_counter == 0:
                splits.append(curr_split.strip())
                curr_split = ""
                continue
            elif c == "(":
                open_counter += 1
            elif c == ")":
                open_counter -= 1
            curr_split += c
        splits.append(curr_split.strip())
        return splits