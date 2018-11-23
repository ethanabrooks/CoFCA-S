import argparse
from collections.__init__ import namedtuple
from contextlib import contextmanager
from itertools import filterfalse
from pathlib import Path
import re
import tempfile
from typing import List, Tuple
from xml.etree import ElementTree as ET

from gym import spaces
from gym.spaces import Box
import numpy as np
import tensorflow as tf
from utils.utils import parametric_relu


def make_box(*tuples: Tuple[float, float]):
    low, high = map(np.array, zip(*[(map(float, m)) for m in tuples]))
    return spaces.Box(low=low, high=high, dtype=np.float32)


def parse_space(dim: int):
    def _parse_space(arg: str):
        regex = re.compile('\((-?[\.\d]+),(-?[\.\d]+)\)')
        matches = regex.findall(arg)
        if len(matches) != dim:
            raise argparse.ArgumentTypeError(
                f'Arg {arg} must have {dim} substrings '
                f'matching pattern {regex}.')
        return make_box(*matches)

    return _parse_space


def parse_vector(length: int, delim: str):
    def _parse_vector(arg: str):
        vector = tuple(map(float, arg.split(delim)))
        if len(vector) != length:
            raise argparse.ArgumentError(
                f'Arg {arg} must include {length} float values'
                f'delimited by "{delim}".')
        return vector

    return _parse_vector


def cast_to_int(arg: str):
    return int(float(arg))


ACTIVATIONS = dict(
    relu=tf.nn.relu,
    leaky=tf.nn.leaky_relu,
    elu=tf.nn.elu,
    selu=tf.nn.selu,
    prelu=parametric_relu,
    sigmoid=tf.sigmoid,
    tanh=tf.tanh,
    none=None,
)


def parse_activation(arg: str):
    return ACTIVATIONS[arg]


def put_in_xml_setter(arg: str):
    setters = [XMLSetter(*v.split(',')) for v in arg]
    mirroring = [XMLSetter(p.replace('_l_', '_r_'), v)
                 for p, v in setters if '_l_' in p] \
                + [XMLSetter(p.replace('_r_', '_l_'), v)
                   for p, v in setters if '_r_' in p]
    return [s._replace(path=s.path) for s in setters + mirroring]


XMLSetter = namedtuple('XMLSetter', 'path value')


@contextmanager
def mutate_xml(changes: List[XMLSetter], dofs: List[str], goal_space: Box,
               n_blocks: int, xml_filepath: Path):
    def rel_to_abs(path: Path):
        return Path(xml_filepath.parent, path)

    def mutate_tree(tree: ET.ElementTree):

        worldbody = tree.getroot().find("./worldbody")
        rgba = [
            "0 1 0 1",
            "0 0 1 1",
            "0 1 1 1",
            "1 0 0 1",
            "1 0 1 1",
            "1 1 0 1",
            "1 1 1 1",
        ]

        if worldbody:
            for i in range(n_blocks):
                pos = ' '.join(map(str, goal_space.sample()))
                name = f'block{i}'

                body = ET.SubElement(
                    worldbody, 'body', attrib=dict(name=name, pos=pos))
                ET.SubElement(
                    body,
                    'geom',
                    attrib=dict(
                        name=name,
                        type='box',
                        mass='1',
                        size=".05 .025 .017",
                        rgba=rgba[i],
                        condim='6',
                        solimp="0.99 0.99 "
                        "0.01",
                        solref='0.01 1'))
                ET.SubElement(
                    body, 'freejoint', attrib=dict(name=f'block{i}joint'))

        for change in changes:
            parent = re.sub('/[^/]*$', '', change.path)
            element_to_change = tree.find(parent)
            if isinstance(element_to_change, ET.Element):
                print('setting', change.path, 'to', change.value)
                name = re.search('[^/]*$', change.path)[0]
                element_to_change.set(name, change.value)

        for actuators in tree.iter('actuator'):
            for actuator in list(actuators):
                if actuator.get('joint') not in dofs:
                    print('removing', actuator.get('name'))
                    actuators.remove(actuator)
        for body in tree.iter('body'):
            for joint in body.findall('joint'):
                if not joint.get('name') in dofs:
                    print('removing', joint.get('name'))
                    body.remove(joint)

        parent = Path(temp[xml_filepath].name).parent

        for include_elt in tree.findall('*/include'):
            original_abs_path = rel_to_abs(include_elt.get('file'))
            tmp_abs_path = Path(temp[original_abs_path].name)
            include_elt.set('file', str(tmp_abs_path.relative_to(parent)))

        for compiler in tree.findall('compiler'):
            abs_path = rel_to_abs(compiler.get('meshdir'))
            compiler.set('meshdir', str(abs_path))

        return tree

    included_files = [
        rel_to_abs(e.get('file'))
        for e in ET.parse(xml_filepath).findall('*/include')
    ]

    temp = {
        path: tempfile.NamedTemporaryFile()
        for path in (included_files + [xml_filepath])
    }
    try:
        for path, f in temp.items():
            tree = ET.parse(path)
            mutate_tree(tree)
            tree.write(f)
            f.flush()

        yield Path(temp[xml_filepath].name)
    finally:
        for f in temp.values():
            f.close()


def parse_groups(parser: argparse.ArgumentParser):
    args = parser.parse_args()

    def is_optional(group):
        return group.title == 'optional arguments'

    def parse_group(group):
        # noinspection PyProtectedMember
        return {
            a.dest: getattr(args, a.dest, None)
            for a in group._group_actions
        }

    # noinspection PyUnresolvedReferences,PyProtectedMember
    groups = [
        g for g in parser._action_groups if g.title != 'positional arguments'
    ]
    optional = filter(is_optional, groups)
    not_optional = filterfalse(is_optional, groups)

    kwarg_dicts = {group.title: parse_group(group) for group in not_optional}
    kwargs = (parse_group(next(optional)))
    del kwargs['help']
    return {**kwarg_dicts, **kwargs}
