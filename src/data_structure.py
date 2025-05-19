from dataclasses import dataclass

from ordered_set import OrderedSet


@dataclass(frozen=True, eq=True)
class Predicate:
    id: int
    name: str

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Predicate):
            return self.name == other.name  # Equality based only on 'name'
        return False


@dataclass(frozen=True, eq=True)
class Fact:
    id: int
    subject: str
    text: str
    fol: Predicate
    str_fol: str
    negation: bool

    def __hash__(self):
        return hash((self.text, self.negation))

    def __eq__(self, other):
        if isinstance(other, Fact):
            return self.text == other.text


@dataclass(frozen=True, eq=True)
class Rule:
    id: int
    text: str
    fol: str
    str_fol: str

    def __hash__(self):
        return hash(self.text)

    def __eq__(self, other):
        if isinstance(other, Rule):
            return self.text == other.text


@dataclass
class KB:
    predicates: OrderedSet[Predicate]
    all_facts: list[Fact]
    all_rules: list[Rule]
    context_facts: list[Fact]
    context_rules: list[Rule]
    context: list[Fact | Rule]
    conclusion: Fact | Rule
    background_story: str
    subject_name: str
    subject_category: str
    keyword: str
