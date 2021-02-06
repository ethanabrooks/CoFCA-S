from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from enum import Enum, unique, auto
from typing import List, Generator, Optional

# noinspection PyShadowingBuiltins
from numpy.random.mtrand import RandomState

MAX_SUBTASK = 10


def sample(random, _min, _max, p=0.5):
    return min(_min + random.geometric(p) - 1, _max)


@dataclass(frozen=True)
class State:
    action: Optional[int]
    agent_pointer: int
    env_pointer: int
    success: bool
    failure: bool
    time_remaining: int


@dataclass(frozen=True)
class Line:
    pass


@dataclass(frozen=True)
class Expression:
    @abstractmethod
    def __iter__(self) -> Generator[Line, None, None]:
        pass

    def __len__(self) -> int:
        return sum(1 for _ in self)

    @abstractmethod
    def is_complete(self) -> bool:
        pass

    @staticmethod
    def multi_line_expressions() -> List[type]:
        return [Sequence, WhileLoop, IfCondition, IfElseCondition]

    @abstractmethod
    def next_subtask_if_not_complete(self, passing: bool) -> Optional["Subtask"]:
        pass

    def next_subtask(self, passing: bool) -> Optional["Subtask"]:
        if self.is_complete():
            return None
        return self.next_subtask_if_not_complete(passing)

    @classmethod
    def random(cls, length: int, rng: RandomState) -> "Expression":
        if length == 1:
            return Subtask.random(rng)
        else:

            def short_enough(t: type):
                assert issubclass(t, MultiLineExpression)
                return t.required_lines() <= length

            expr = rng.choice([*filter(short_enough, cls.multi_line_expressions())])
            assert issubclass(expr, MultiLineExpression)
            return expr.random(length, rng)

    @abstractmethod
    def reset(self) -> "Expression":
        pass

    @abstractmethod
    def update(self, passing: bool) -> "Expression":
        pass


@dataclass(frozen=True)
class MultiLineExpression(Expression):
    @staticmethod
    @abstractmethod
    def random(length: int, rng: RandomState) -> "Expression":
        pass

    @staticmethod
    @abstractmethod
    def required_lines() -> int:
        pass


@dataclass(frozen=True)
class Sequence(MultiLineExpression):
    expr1: Expression
    expr2: Expression

    def __iter__(self) -> Generator[Line, None, None]:
        for expr in [self.expr1, self.expr2]:
            yield from expr

    def is_complete(self) -> bool:
        return self.expr1.is_complete() and self.expr2.is_complete()

    def next_subtask_if_not_complete(self, passing: bool) -> Optional["Subtask"]:
        subtask = self.expr1.next_subtask(passing)
        if subtask is not None:
            return subtask
        return self.expr2.next_subtask(passing)

    @staticmethod
    def random(length: int, rng: RandomState) -> "Expression":
        n1 = rng.randint(1, length)  # {1,...,length-1}
        n2 = length - n1
        return Sequence(Expression.random(n1, rng), Expression.random(n2, rng))

    @abstractmethod
    def reset(self) -> "Sequence":
        return replace(self, expr1=self.expr1.reset(), expr2=self.expr2.reset())

    @staticmethod
    def required_lines() -> int:
        return 2

    def update(self, passing: bool) -> "Sequence":
        if not self.expr1.is_complete():
            return replace(self, expr1=self.expr1.update(passing))
        else:
            return replace(self, expr2=self.expr2.update(passing))


@dataclass(frozen=True)
class Subtask(Expression, Line):
    id: int
    complete: bool = False

    def __iter__(self) -> Generator[Line, None, None]:
        yield self

    def is_complete(self) -> bool:
        return self.complete

    def next_subtask_if_not_complete(self, passing: bool) -> Optional["Subtask"]:
        return self

    @staticmethod
    def random(rng: RandomState) -> "Subtask":
        return Subtask(rng.randint(MAX_SUBTASK))

    def reset(self) -> "Subtask":
        return replace(self, complete=False)

    def update(self, passing: bool) -> "Expression":
        return replace(self, complete=True)


@dataclass(frozen=True)
class While(Line):
    pass


@unique
class Predicate(Enum):
    PASSED = auto()
    FAILED = auto()
    NOT_TESTED = auto()


@dataclass(frozen=True)
class Condition(MultiLineExpression, ABC):
    predicate: Predicate

    @staticmethod
    def required_lines() -> int:
        return 3

    def tested(self):
        return self.predicate is not Predicate.NOT_TESTED


@dataclass(frozen=True)
class SingleExpressionCondition(Condition, ABC):
    expr: Expression

    @abstractmethod
    def needs_test(self) -> bool:
        pass

    def reset(self) -> "SingleExpressionCondition":
        return replace(self, expr=self.expr.reset(), predicate=Predicate.NOT_TESTED)


@dataclass(frozen=True)
class WhileLoop(SingleExpressionCondition):
    def __iter__(self) -> Generator[Line, None, None]:
        yield While()
        yield from self.expr
        yield EndWhile()

    def is_complete(self) -> bool:
        return self.predicate is Predicate.FAILED

    def needs_test(self) -> bool:
        return not self.tested() or self.expr.is_complete()

    def next_subtask_if_not_complete(self, passing: bool) -> Optional["Subtask"]:
        if self.needs_test() and not passing:
            return None
        return self.expr.next_subtask(passing)

    @staticmethod
    def random(length: int, rng: RandomState) -> "Expression":
        expr = Expression.random(length - 2, rng)  # 2 for While and EndWhile
        return WhileLoop(expr=expr, predicate=Predicate.NOT_TESTED)

    def update(self, passing: bool) -> "Condition":
        if self.needs_test():
            if passing:
                expr = self.expr.update(passing)
                predicate = Predicate.PASSED
            else:
                expr = self.expr
                predicate = Predicate.FAILED
        else:
            expr = self.expr.update(passing)
            predicate = self.predicate
        if expr.is_complete():
            expr = expr.reset()
            predicate = Predicate.NOT_TESTED

        # inside loop
        return replace(self, expr=expr, predicate=predicate)


@dataclass(frozen=True)
class EndWhile(Line):
    pass


@dataclass(frozen=True)
class IfCondition(SingleExpressionCondition):
    def __iter__(self) -> Generator[Line, None, None]:
        yield If()
        yield from self.expr
        yield EndIf()

    def is_complete(self) -> bool:
        return not self.predicate or self.expr.is_complete()

    def needs_test(self) -> bool:
        return not self.tested()

    def next_subtask_if_not_complete(self, passing: bool) -> Optional["Subtask"]:
        if self.needs_test() and not passing or self.predicate is Predicate.FAILED:
            return None
        return self.expr.next_subtask(passing)

    @staticmethod
    def random(length: int, rng: RandomState) -> "Expression":
        return IfCondition(
            expr=Expression.random(length - 2, rng), predicate=Predicate.NOT_TESTED
        )

    def update(self, passing: bool) -> "Condition":
        if self.needs_test():
            if passing:
                expr = self.expr.update(passing)
                return replace(self, predicate=Predicate.PASSED, expr=expr)
            else:
                return replace(self, predicate=Predicate.FAILED)
        # inside loop
        return replace(self, expr=self.expr.update(passing))


@dataclass(frozen=True)
class If(Line):
    pass


@dataclass(frozen=True)
class EndIf(Line):
    pass


@dataclass(frozen=True)
class IfElseCondition(Condition):
    expr1: Expression
    expr2: Expression

    def __iter__(self) -> Generator[Line, None, None]:
        yield If()
        yield from self.expr1
        yield Else()
        yield from self.expr2
        yield EndIf()

    def is_complete(self) -> bool:
        return self.expr1.is_complete() or self.expr2.is_complete()

    def next_subtask_if_not_complete(self, passing: bool) -> Optional["Subtask"]:
        if self.predicate is Predicate.NOT_TESTED:
            return (self.expr1 if passing else self.expr2).next_subtask(passing)
        if self.predicate is Predicate.PASSED:
            return self.expr1.next_subtask(passing)
        if self.predicate is Predicate.FAILED:
            return self.expr2.next_subtask(passing)
        raise RuntimeError

    @staticmethod
    def random(length: int, rng: RandomState) -> "Expression":
        expr1_length = rng.randint(1, length - 3)
        # {1,...,length-4} (4 for If, Else, Expr2, EndIf)

        expr2_length = length - expr1_length
        return IfElseCondition(
            expr1=Expression.random(expr1_length, rng),
            expr2=Expression.random(expr2_length, rng),
            predicate=Predicate.NOT_TESTED,
        )

    @staticmethod
    def required_lines() -> int:
        return 5

    def reset(self) -> "Expression":
        pass

    def update(self, passing: bool) -> "Expression":
        if self.predicate is Predicate.NOT_TESTED:
            if passing:
                expr1 = self.expr1.update(passing)
                return replace(self, predicate=Predicate.PASSED, expr1=expr1)
            else:
                expr2 = self.expr2.update(passing)
                return replace(self, predicate=Predicate.FAILED, expr2=expr2)
        if self.predicate is Predicate.PASSED:
            return replace(self, expr1=self.expr1.update(passing))
        if self.predicate is Predicate.FAILED:
            return replace(self, expr2=self.expr2.update(passing))
        raise RuntimeError


@dataclass(frozen=True)
class Else(Line):
    pass


#
# @dataclass(frozen=True)
# class Action(data_types.RawAction):
#     @staticmethod
#     def parse(*xs) -> "Action":
#         delta, gate, ptr, extrinsic = xs
#         return Action(delta, gate, ptr, extrinsic)
#
#     @property
#     def is_op(self):
#         return self.extrinsic is not None


if __name__ == "__main__":
    from gym.utils.seeding import np_random
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("length", type=int)
    parser.add_argument("seed", type=int)
    args = parser.parse_args()

    rng, _ = np_random(args.seed)
    for cls in Expression.multi_line_expressions():
        assert issubclass(cls, MultiLineExpression)
        if cls.required_lines() <= args.length:
            ex = cls.random(args.length, rng)
            print("ex")
            print(ex)
            print("ex.next_subtask(True)")
            print(ex.next_subtask(True))
            print("ex.next_subtask(False)")
            print(ex.next_subtask(False))
            ex_true = ex.update(True)
            print("ex_true")
            print(ex_true)
            print("ex_true.next_subtask(True)")
            print(ex_true.next_subtask(True))
            print("ex_true.next_subtask(False)")
            print(ex_true.next_subtask(False))
            ex_false = ex.update(False)
            print("ex_false")
            print(ex_false)
            breakpoint()
            print("ex_false.next_subtask(False)")
            print(ex_false.next_subtask(False))
            print("ex_false.next_subtask(False)")
            print(ex_false.next_subtask(False))
            breakpoint()
            ex.update(True)
