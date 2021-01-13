import numpy as np


class Templates:
    def __init__(self, shape_model=None, cell_size=6):
        self._sizes = None
        self._templates = None
        if not shape_model:
            shape_model = np.zeros([20, 10])
            shape_model[2:4, 4:6] = 1
            shape_model[4:11, 2:8] = 2
            shape_model[11:18, 3:7] = 3
            self._shape_model = shape_model
        else:
            self._shape_model = shape_model
            self._cell_size = cell_size

    def generate_sizes(self, height_max=4, width_max=3) -> None:
        """
        S = {(w, h) | w ≤ wm, h ≤ hm, w, h ∈ N+}

        Templates size ranging from 1 × 2 to 4 × 3 cells, we obtain 266 templates

        :param height_max: maximum height of templates
        :param width_max: maximum width of templates
        :return: None
        """
        sizes = []
        for h in range(1, height_max + 1):
            for w in range(1, width_max + 1):
                sizes.append((w, h))
        self._sizes = sizes[1:]

    def _remove_dublicates(self):
        remove = []

        for l1, t1 in enumerate(self._templates):
            for l2, t2 in enumerate(self._templates[l1+1:]):

                if t1[0] == t2[0] and t1[1] == t2[1]:
                    _, _, size1, W1 = t1
                    _, _, size2, W2 = t2
                    w1, h1 = size1
                    w2, h2 = size2
                    wmax = max([w1, w2])
                    hmax = max([h1, h2])

                    w1p = np.zeros([hmax, wmax])
                    w2p = np.zeros([hmax, wmax])
                    w1p[:h1, :w1] = W1
                    w2p[:h2, :w2] = W2

                    if np.sum(np.abs(w1p - w2p)) == 0:
                        remove.append(l1)
                        break

        indices = [x for x in range(len(self._templates)) if x not in remove]
        self._templates = self._templates[indices]

    def _shift_templates(self):
        new_templates = []
        for t in self._templates:
            x, y, size, W = t

            if y < self._shape_model.shape[0] - 1:
                new_templates.append((x, y+1, size, W))

            if y > 0:
                new_templates.append((x, y-1, size, W))

            if x < self._shape_model.shape[1] - 1:
                new_templates.append((x+1, y, size, W))

            if x > 0:
                new_templates.append((x-1, y, size, W))

        new_templates = np.asarray(new_templates, dtype=object)
        self._templates = np.concatenate((self._templates, new_templates), axis=0)

    def _normalize_templates(self):
        for id, t in enumerate(self._templates):
            x, y, size, W = t

            W1 = np.copy(W)
            W2 = np.copy(W)

            W1[W1 != 1] = 0
            W2[W2 != -1] = 0

            s1 = np.sum(W1)
            s2 = np.sum(-W2)

            if s2:
                self._templates[id] = (x, y, size, np.copy(W1/s1 + W2/s2))
            else:
                self._templates[id] = (x, y, size, np.copy(W1/s1))

    def generate_templates(self):
        """
        T = {(x, y, s, W) | x, y ∈ N, s ∈ S, W ∈ R2}

        x: indicate the location x of a template
        y: indicate the location y of a template
        s: size of the template with format (width, height) in terms of covered cells
        W: weight matrix that is determined according to the matrix L of labels for all cells
        """

        templates = []

        for size in self._sizes:
            width = size[0]
            height = size[1]
            for y in range(self._shape_model.shape[0] - height):
                for x in range(self._shape_model.shape[1] - width):
                    matrix = np.copy(self._shape_model[y: y+height, x: x+width])
                    unique = np.unique(matrix)

                    if len(unique) > 1:
                        if len(unique) == 2:
                            l1 = matrix == unique[0]
                            l2 = matrix == unique[1]

                            matrix[l1] = 1
                            matrix[l2] = 0
                            templates.append((x, y, size, matrix))

                        else:
                            l1 = matrix == unique[0]
                            l2 = matrix == unique[1]
                            l3 = matrix == unique[2]

                            matrix[l1] = -1
                            matrix[l2] = 0
                            matrix[l3] = 1
                            templates.append((x, y, size, matrix))

                            matrix[l1] = 1
                            matrix[l2] = -1
                            matrix[l3] = 0
                            templates.append((x, y, size, matrix))

                            matrix[l1] = 0
                            matrix[l2] = 1
                            matrix[l3] = -1
                            templates.append((x, y, size, matrix))

        self._templates = np.asarray(templates, dtype=object)
        self._remove_dublicates()
        self._shift_templates()
        self._normalize_templates()


        print(f'Created {len(self._templates)} templates')

        return self._templates