from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import List, Generator, Optional, Union

from colored import fg
from gym.spaces import Discrete
from numpy.random.mtrand import RandomState

from data_types import RawAction
from utils import RESET

NUM_SUBTASKS = 10


@dataclass(frozen=True)
class State:
    action: "Action"
    agent_pointer: int
    env_pointer: int
    success: bool
    wrong_move: bool
    condition_bit: bool
    time_remaining: int


@dataclass(frozen=True)
class Line:
    def __eq__(self, other):
        return type(self) == type(other)

    @staticmethod
    def space() -> Discrete:
        return Discrete(len(NonSubtaskLines) + NUM_SUBTASKS)

    @classmethod
    def to_int(cls) -> int:
        # noinspection PyTypeChecker
        return NonSubtaskLines.index(cls)


@dataclass(frozen=True)
class If(Line):
    pass


@dataclass(frozen=True)
class Else(Line):
    pass


@dataclass(frozen=True)
class EndIf(Line):
    pass


@dataclass(frozen=True)
class While(Line):
    pass


@dataclass(frozen=True)
class EndWhile(Line):
    pass


@dataclass(frozen=True)
class Pad(Line):
    pass


@dataclass(frozen=True)
class Expression:
    @abstractmethod
    def __iter__(self) -> Generator[Line, None, None]:
        raise NotImplementedError

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __str__(self):
        return "\n".join(self.strings())

    @staticmethod
    def multi_line_expressions() -> List[type]:
        return [WhileLoop, IfCondition, IfElseCondition, Sequence]

    @abstractmethod
    def complete(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def inside_passing_if(self) -> "IfCondition":
        raise NotImplementedError

    @abstractmethod
    def inside_passing_while(self) -> "WhileLoop":
        raise NotImplementedError

    @abstractmethod
    def as_first_expression_of_passing_if_else(
        self, expr2: "IncompleteExpression"
    ) -> "IfElseCondition":
        raise NotImplementedError

    @abstractmethod
    def as_second_expression_of_failing_if_else(
        self, expr1: "IncompleteExpression"
    ) -> "IfElseCondition":
        raise NotImplementedError

    @abstractmethod
    def followed_by(self, expr: "Expression") -> "Sequence":
        raise NotImplementedError

    @abstractmethod
    def preceded_by_complete(
        self, expr: "CompleteExpression"
    ) -> Union[
        "SequenceWithUnpredicatedExpr2",
        "SequenceWithReadyExpr2",
        "CompleteSequence",
    ]:
        raise NotImplementedError

    @abstractmethod
    def predicated(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> Union["UnpredicatedExpression", "ReadyExpression"]:
        raise NotImplementedError

    @classmethod
    def random(
        cls, length: int, rng: RandomState, max_depth: int = float("inf")
    ) -> Union["UnpredicatedExpression", "ReadyExpression"]:
        if length == 1:
            return Subtask.random(rng)
        elif length > 1:

            def possible(t: type):
                assert issubclass(t, MultiLineExpression)
                return t.required_lines() <= length and t.required_depth() <= max_depth

            expr = rng.choice([*filter(possible, cls.multi_line_expressions())])
            assert issubclass(expr, MultiLineExpression)
            return expr.random(length=length, rng=rng, max_depth=max_depth)
        else:
            raise RuntimeError

    @abstractmethod
    def set_predicate(
        self, passing: bool
    ) -> Union["ReadyExpression", "CompleteExpression"]:
        raise NotImplementedError

    @abstractmethod
    def strings(self) -> Generator[str, None, None]:
        raise NotImplementedError

    @abstractmethod
    def to_ints(self) -> Generator[int, None, None]:
        raise NotImplementedError


@dataclass(frozen=True)
class IncompleteExpression(Expression, ABC):
    def complete(self) -> bool:
        return False


@dataclass(frozen=True)
class PredicatedExpression(Expression, ABC):
    def predicated(self) -> bool:
        return True


@dataclass(frozen=True)
class UnpredicatedExpression(IncompleteExpression, ABC):
    def predicated(self) -> bool:
        return False

    def followed_by(
        self, expr: "UnpredicatedExpression"
    ) -> "SequenceWithUnpredicatedExpr1":
        return SequenceWithUnpredicatedExpr1(self, expr)

    def preceded_by_complete(self, expr: "CompleteExpression") -> "Sequence":
        return SequenceWithUnpredicatedExpr2(expr, self)

    def inside_passing_if(self) -> "PassingUnpredicatedIfCondition":
        return PassingUnpredicatedIfCondition(self)

    def inside_passing_while(self) -> "PassingWhileLoopWithUnpredicatedExpr":
        return PassingWhileLoopWithUnpredicatedExpr(self)

    def as_first_expression_of_passing_if_else(
        self, expr2: IncompleteExpression
    ) -> "IfElseCondition":
        return PassingIfElseConditionWithUnpredicatedExpr(self, expr2)

    def as_second_expression_of_failing_if_else(
        self, expr1: IncompleteExpression
    ) -> "IfElseCondition":
        return FailingIfElseConditionWithUnpredicatedExpr(expr1, self)

    @abstractmethod
    def _strings(self) -> Generator[str, None, None]:
        raise NotImplementedError

    def strings(self) -> Generator[str, None, None]:
        yield from self._strings()


@dataclass(frozen=True)
class ReadyExpression(IncompleteExpression, PredicatedExpression):
    @abstractmethod
    def advance(self) -> Union["IncompleteExpression", "CompleteExpression"]:
        raise NotImplementedError

    @abstractmethod
    def subtask(self) -> "Subtask":
        raise NotImplementedError

    def followed_by(self, expr: "ReadyExpression") -> "SequenceWithReadyExpr1":
        return SequenceWithReadyExpr1(self, expr)

    def inside_passing_while(self) -> "PassingWhileLoopWithReadyExpr":
        return PassingWhileLoopWithReadyExpr(self)

    def preceded_by_complete(self, expr: "CompleteExpression") -> "Sequence":
        return SequenceWithReadyExpr2(expr, self)

    def inside_passing_if(self) -> "PassingReadyIfCondition":
        return PassingReadyIfCondition(self)

    def set_predicate(
        self, passing: bool
    ) -> Union["ReadyExpression", "CompleteExpression"]:
        return self

    def as_first_expression_of_passing_if_else(
        self, expr2: IncompleteExpression
    ) -> "PassingIfElseConditionWithReadyExpr":
        return PassingIfElseConditionWithReadyExpr(self, expr2)

    def as_second_expression_of_failing_if_else(
        self, expr1: IncompleteExpression
    ) -> "FailingIfElseConditionWithReadyExpr":
        return FailingIfElseConditionWithReadyExpr(expr1, self)

    @abstractmethod
    def _strings(self) -> Generator[str, None, None]:
        raise NotImplementedError

    def strings(self) -> Generator[str, None, None]:
        yield from self._strings()


@dataclass(frozen=True)
class CompleteExpression(PredicatedExpression, ABC):
    def followed_by(
        self, expr: "Expression"
    ) -> Union[
        "SequenceWithUnpredicatedExpr2",
        "SequenceWithReadyExpr2",
        "CompleteSequence",
    ]:
        return expr.preceded_by_complete(self)

    def preceded_by_complete(self, expr: "CompleteExpression") -> "CompleteSequence":
        return CompleteSequence(expr, self)

    def complete(self) -> bool:
        return True

    def inside_passing_if(self) -> "IfCondition":
        return PassingCompleteIfCondition(self)

    def as_first_expression_of_passing_if_else(
        self, expr2: IncompleteExpression
    ) -> "CompletePassingIfElseCondition":
        return CompletePassingIfElseCondition(self, expr2)

    def as_second_expression_of_failing_if_else(
        self, expr1: IncompleteExpression
    ) -> "CompleteFailingIfElseCondition":
        return CompleteFailingIfElseCondition(expr1, self)

    def inside_passing_while(self) -> "UnpredicatedWhileLoop":
        return UnpredicatedWhileLoop(self.reset())

    def set_predicate(
        self, passing: bool
    ) -> Union["ReadyExpression", "CompleteExpression"]:
        return self

    @abstractmethod
    def _strings(self) -> Generator[str, None, None]:
        raise NotImplementedError

    def strings(self) -> Generator[str, None, None]:
        for string in self._strings():
            yield f"{fg('dark_gray')}{string}{RESET}"


@dataclass(frozen=True)
class Subtask(PredicatedExpression, Line, ABC):
    id: int

    def __eq__(self, other):
        return super().__eq__(other) and self.id == other.id

    @staticmethod
    def random(rng: RandomState) -> "IncompleteSubtask":
        return IncompleteSubtask(id=rng.randint(NUM_SUBTASKS))

    def reset(self) -> "IncompleteExpression":
        return IncompleteSubtask(id=self.id)

    def _strings(self) -> Generator[str, None, None]:
        yield f"Subtask {self.id}"

    def to_int(self) -> int:
        return len(NonSubtaskLines) + self.id

    def to_ints(self) -> Generator[int, None, None]:
        yield self.to_int()


@dataclass(frozen=True)
class IncompleteSubtask(Subtask, ReadyExpression):
    def __iter__(self) -> Generator[Line, None, None]:
        yield self

    def preceded_by_complete(self, expr: "CompleteExpression") -> "Sequence":
        return SequenceWithReadyExpr2(expr, self)

    def advance(self) -> "CompleteSubtask":
        return CompleteSubtask(id=self.id)

    def subtask(self) -> "Subtask":
        return self


@dataclass(frozen=True)
class CompleteSubtask(Subtask, CompleteExpression):
    def __iter__(self) -> Generator[Line, None, None]:
        yield self


@dataclass(frozen=True)
class MultiLineExpression(Expression):
    @staticmethod
    @abstractmethod
    def required_lines() -> int:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def required_depth() -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Sequence(MultiLineExpression, ABC):
    expr1: Expression
    expr2: Expression

    def __iter__(self) -> Generator[Line, None, None]:
        yield from self.expr1
        yield from self.expr2

    def _strings(self) -> Generator[str, None, None]:
        yield from self.expr1.strings()
        yield from self.expr2.strings()

    @staticmethod
    def random(
        length: int, rng: RandomState, max_depth: int = float("inf")
    ) -> "IncompleteExpression":
        n1 = rng.randint(1, length)  # {1,...,length-1}
        n2 = length - n1
        expr1 = MultiLineExpression.random(n1, rng, max_depth)
        expr2 = MultiLineExpression.random(n2, rng, max_depth)
        return expr1.followed_by(expr2)

    @staticmethod
    def required_lines() -> int:
        return 2

    @staticmethod
    def required_depth() -> int:
        return 0

    def reset(self) -> "SequenceWithUnpredicatedExpr1":
        return self.expr1.reset().followed_by(self.expr2.reset())

    def to_ints(self) -> Generator[int, None, None]:
        yield from self.expr1.to_ints()
        yield from self.expr2.to_ints()


@dataclass(frozen=True)
class SequenceWithUnpredicatedExpr1(Sequence, UnpredicatedExpression):
    expr1: UnpredicatedExpression
    expr2: UnpredicatedExpression

    @staticmethod
    def required_lines() -> int:
        return 2

    @staticmethod
    def required_depth() -> int:
        return 1

    def set_predicate(self, passing: bool) -> "SequenceWithReadyExpr1":
        return self.expr1.set_predicate(passing).followed_by(
            self.expr2.set_predicate(passing)
        )


@dataclass(frozen=True)
class SequenceWithReadyExpr1(Sequence, ReadyExpression):
    expr1: ReadyExpression
    expr2: Expression

    def advance(self) -> "Sequence":
        return self.expr1.advance().followed_by(self.expr2)

    def subtask(self) -> "Subtask":
        return self.expr1.subtask()


@dataclass(frozen=True)
class SequenceWithUnpredicatedExpr2(Sequence, UnpredicatedExpression):
    expr1: CompleteExpression
    expr2: UnpredicatedExpression

    def set_predicate(
        self, passing: bool
    ) -> Union["SequenceWithReadyExpr2", "CompleteSequence"]:
        return self.expr1.followed_by(self.expr2.set_predicate(passing))


@dataclass(frozen=True)
class SequenceWithReadyExpr2(Sequence, ReadyExpression):
    expr1: CompleteExpression
    expr2: ReadyExpression

    def advance(self) -> "Sequence":
        return self.expr1.followed_by(self.expr2.advance())

    def subtask(self) -> "Subtask":
        return self.expr2.subtask()


@dataclass(frozen=True)
class CompleteSequence(Sequence, CompleteExpression):
    expr1: CompleteExpression
    expr2: CompleteExpression


@dataclass(frozen=True)
class IfCondition(MultiLineExpression, ABC):
    expr: Expression

    def __iter__(self) -> Generator[Line, None, None]:
        yield If()
        yield from self.expr
        yield EndIf()

    @staticmethod
    def random(
        length: int, rng: RandomState, max_depth: int = float("inf")
    ) -> "IncompleteExpression":
        return UnpredicatedIfCondition(
            MultiLineExpression.random(length - 2, rng, max_depth=max_depth - 1)
        )

    @staticmethod
    def required_lines() -> int:
        return 3

    @staticmethod
    def required_depth() -> int:
        return 1

    def reset(self) -> Union["UnpredicatedExpression", "ReadyExpression"]:
        return UnpredicatedIfCondition(self.expr.reset())

    def to_ints(self) -> Generator[int, None, None]:
        yield If.to_int()
        yield from self.expr.to_ints()
        yield EndIf.to_int()


@dataclass(frozen=True)
class UnpredicatedIfCondition(IfCondition, UnpredicatedExpression):
    expr: IncompleteExpression

    def _strings(self) -> Generator[str, None, None]:
        yield "If"
        for string in self.expr.strings():
            yield f"  {string}"
        yield "EndIf"

    def set_predicate(
        self, passing: bool
    ) -> Union["ReadyExpression", "CompleteExpression"]:
        if passing:
            return self.expr.set_predicate(passing).inside_passing_if()
        else:
            return FailingIfCondition(self.expr)


@dataclass(frozen=True)
class PassingIfCondition(IfCondition, PredicatedExpression, ABC):
    def _strings(self) -> Generator[str, None, None]:
        yield "If (passing)"
        for string in self.expr.strings():
            yield f"  {string}"
        yield "EndIf"

    def complete(self) -> bool:
        return self.expr.complete()


@dataclass(frozen=True)
class PassingUnpredicatedIfCondition(PassingIfCondition, UnpredicatedExpression):
    expr: UnpredicatedExpression

    def set_predicate(
        self, passing: bool
    ) -> Union["ReadyExpression", "CompleteExpression"]:
        return self.expr.set_predicate(passing).inside_passing_if()


@dataclass(frozen=True)
class PassingReadyIfCondition(PassingIfCondition, ReadyExpression):
    expr: ReadyExpression

    def advance(self) -> "Expression":
        return self.expr.advance().inside_passing_if()

    def subtask(self) -> "Subtask":
        return self.expr.subtask()


@dataclass(frozen=True)
class PassingCompleteIfCondition(PassingIfCondition, CompleteExpression):
    expr: CompleteExpression

    def reset(self) -> "IncompleteExpression":
        return UnpredicatedIfCondition(self.expr.reset())


@dataclass(frozen=True)
class FailingIfCondition(IfCondition, CompleteExpression):
    expr: IncompleteExpression

    def _strings(self) -> Generator[str, None, None]:
        yield "If (failing)"
        for string in self.expr.strings():
            yield f"  {string}"
        yield "EndIf"

    def reset(self) -> "IncompleteExpression":
        return UnpredicatedIfCondition(self.expr.reset())


@dataclass(frozen=True)
class IfElseCondition(MultiLineExpression, ABC):
    expr1: Expression
    expr2: Expression

    def _strings(self) -> Generator[str, None, None]:
        yield "If"
        for string in self.expr1.strings():
            yield f"  {string}"
        yield "Else"
        for string in self.expr2.strings():
            yield f"  {string}"
        yield "EndIf"

    def __iter__(self) -> Generator[Line, None, None]:
        yield If()
        yield from self.expr1
        yield Else()
        yield from self.expr2
        yield EndIf()

    @staticmethod
    def random(
        length: int, rng: RandomState, max_depth: int = float("inf")
    ) -> "Expression":
        expr1_length = rng.randint(1, length - 3)
        # {1,...,length-4} (4 for If, Else, Expr2, EndIf)

        expr2_length = length - expr1_length - 3
        return UnpredicatedIfElseCondition(
            expr1=Expression.random(expr1_length, rng, max_depth=max_depth - 1),
            expr2=Expression.random(expr2_length, rng, max_depth=max_depth - 1),
        )

    @staticmethod
    def required_lines() -> int:
        return 5

    @staticmethod
    def required_depth() -> int:
        return 1

    def reset(self) -> Union["UnpredicatedExpression", "ReadyExpression"]:
        return UnpredicatedIfElseCondition(self.expr1.reset(), self.expr2.reset())

    def to_ints(self) -> Generator[int, None, None]:
        yield If.to_int()
        yield from self.expr1.to_ints()
        yield Else.to_int()
        yield from self.expr2.to_ints()
        yield EndIf.to_int()


@dataclass(frozen=True)
class UnpredicatedIfElseCondition(IfElseCondition, UnpredicatedExpression):
    expr1: IncompleteExpression
    expr2: IncompleteExpression

    def set_predicate(
        self, passing: bool
    ) -> Union["ReadyExpression", "CompleteExpression"]:
        if passing:
            expr1 = self.expr1.set_predicate(passing)
            return expr1.as_first_expression_of_passing_if_else(self.expr2)
        else:
            expr2 = self.expr2.set_predicate(passing)
            return expr2.as_second_expression_of_failing_if_else(self.expr1)


@dataclass(frozen=True)
class PassingIfElseCondition(IfElseCondition, ABC):
    expr1: Expression
    expr2: Expression

    def _strings(self) -> Generator[str, None, None]:
        yield "If (passing)"
        for string in self.expr1.strings():
            yield f"  {string}"
        yield "Else"
        for string in self.expr2.strings():
            yield f"  {string}"
        yield "EndIf"


@dataclass(frozen=True)
class PassingIfElseConditionWithUnpredicatedExpr(
    PassingIfElseCondition, UnpredicatedExpression
):
    expr1: UnpredicatedExpression
    expr2: IncompleteExpression

    def set_predicate(
        self, passing: bool
    ) -> Union["ReadyExpression", "CompleteExpression"]:
        expr1 = self.expr1.set_predicate(passing)
        return expr1.as_first_expression_of_passing_if_else(self.expr2)


@dataclass(frozen=True)
class PassingIfElseConditionWithReadyExpr(PassingIfElseCondition, ReadyExpression):
    expr1: ReadyExpression
    expr2: IncompleteExpression

    def advance(self) -> "Expression":
        return self.expr1.advance().as_first_expression_of_passing_if_else(self.expr2)

    def subtask(self) -> "Subtask":
        return self.expr1.subtask()


@dataclass(frozen=True)
class FailingIfElseCondition(IfElseCondition, ABC):
    expr1: Expression
    expr2: Expression

    def _strings(self) -> Generator[str, None, None]:
        yield "If (failing)"
        for string in self.expr1.strings():
            yield f"  {string}"
        yield "Else"
        for string in self.expr2.strings():
            yield f"  {string}"
        yield "EndIf"


@dataclass(frozen=True)
class FailingIfElseConditionWithUnpredicatedExpr(
    FailingIfElseCondition, UnpredicatedExpression
):
    expr1: IncompleteExpression
    expr2: UnpredicatedExpression

    def set_predicate(
        self, passing: bool
    ) -> Union["ReadyExpression", "CompleteExpression"]:
        expr2 = self.expr2.set_predicate(passing)
        return expr2.as_second_expression_of_failing_if_else(self.expr1)


@dataclass(frozen=True)
class FailingIfElseConditionWithReadyExpr(FailingIfElseCondition, ReadyExpression):
    expr1: IncompleteExpression
    expr2: ReadyExpression

    def advance(self) -> "Expression":
        return self.expr2.advance().as_second_expression_of_failing_if_else(self.expr1)

    def subtask(self) -> "Subtask":
        return self.expr2.subtask()


@dataclass(frozen=True)
class CompletePassingIfElseCondition(PassingIfElseCondition, CompleteExpression):
    expr1: Expression
    expr2: Expression


@dataclass(frozen=True)
class CompleteFailingIfElseCondition(FailingIfElseCondition, CompleteExpression):
    expr1: Expression
    expr2: Expression


@dataclass(frozen=True)
class WhileLoop(MultiLineExpression, ABC):
    expr: Expression

    def __iter__(self) -> Generator[Line, None, None]:
        yield While()
        yield from self.expr
        yield EndWhile()

    @staticmethod
    def random(
        length: int, rng: RandomState, max_depth: int = float("inf")
    ) -> "IncompleteExpression":
        return UnpredicatedWhileLoop(
            MultiLineExpression.random(length - 2, rng=rng, max_depth=max_depth - 1)
        )

    @staticmethod
    def required_lines() -> int:
        return 3

    @staticmethod
    def required_depth() -> int:
        return 1

    def reset(self) -> Union["UnpredicatedExpression", "ReadyExpression"]:
        return UnpredicatedWhileLoop(self.expr.reset())

    def _strings(self) -> Generator[str, None, None]:
        yield "While"
        for string in self.expr.strings():
            yield f"  {string}"
        yield "EndWhile"

    def to_ints(self) -> Generator[int, None, None]:
        yield While.to_int()
        yield from self.expr.to_ints()
        yield EndWhile.to_int()


@dataclass(frozen=True)
class UnpredicatedWhileLoop(WhileLoop, UnpredicatedExpression):
    expr: IncompleteExpression

    def set_predicate(
        self, passing: bool
    ) -> Union["ReadyExpression", "CompleteExpression"]:
        if passing:
            predicate = self.expr.set_predicate(passing)
            return predicate.inside_passing_while()
        return FailingWhileLoop(self.expr)


@dataclass(frozen=True)
class PassingWhileLoop(WhileLoop, ABC):
    expr: Expression

    def _strings(self) -> Generator[str, None, None]:
        yield "While (passing)"
        for string in self.expr.strings():
            yield f"  {string}"
        yield "EndWhile"


@dataclass(frozen=True)
class PassingWhileLoopWithUnpredicatedExpr(PassingWhileLoop, UnpredicatedExpression):
    expr: UnpredicatedExpression

    def set_predicate(
        self, passing: bool
    ) -> Union["ReadyExpression", "CompleteExpression"]:
        new = self.expr.set_predicate(passing).inside_passing_while()
        if new.predicated():
            return new
        return new.set_predicate(passing)


@dataclass(frozen=True)
class PassingWhileLoopWithReadyExpr(PassingWhileLoop, ReadyExpression):
    expr: ReadyExpression

    def advance(self) -> "Expression":
        return self.expr.advance().inside_passing_while()

    def subtask(self) -> "Subtask":
        return self.expr.subtask()


@dataclass(frozen=True)
class FailingWhileLoop(WhileLoop, CompleteExpression):
    expr: IncompleteExpression

    def _strings(self) -> Generator[str, None, None]:
        yield "While (failing)"
        for string in self.expr.strings():
            yield f"  {string}"
        yield "EndWhile"


def main(seed, length):
    random, _ = np_random(seed)
    instruction = None
    while True:

        def print_instruction():
            print("predicate", predicate)
            print(instruction)

        if instruction is None:
            instruction = Expression.random(length, random, max_depth=1)
        predicate = random.choice([True, False])
        instruction2 = instruction.set_predicate(predicate)
        if instruction2.complete():
            print_instruction()
            instruction = None
        else:
            print_instruction()
            instruction = instruction2.advance()


@dataclass(frozen=True)
class Action(RawAction):
    @staticmethod
    def parse(
        delta: int = 0, gate: int = 1, pointer: int = 0, extrinsic: int = 0
    ) -> "Action":
        extrinsic = None if extrinsic == 0 else extrinsic - 1
        return Action(delta, gate, pointer, extrinsic)

    def is_op(self) -> bool:
        return self.extrinsic is not None

    def to_ints(self) -> "Action":
        return replace(
            self, extrinsic=0 if self.extrinsic is None else self.extrinsic + 1
        )


NonSubtaskLines = [If, Else, EndIf, While, EndWhile, Pad]

if __name__ == "__main__":
    from gym.utils.seeding import np_random
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("length", type=int)
    parser.add_argument("seed", type=int)
    args = parser.parse_args()
    main(**vars(args))
