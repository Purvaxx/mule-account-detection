import enum
import re
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from .profiler import DatasetProfile, ColumnProfile

class CanonicalConcept(str, enum.Enum):
    TRANSACTION_ID = "TRANSACTION_ID"
    TIMESTAMP = "TIMESTAMP"
    SENDER_NAME = "SENDER_NAME"
    RECEIVER_NAME = "RECEIVER_NAME"
    SENDER_ACCOUNT = "SENDER_ACCOUNT"
    RECEIVER_ACCOUNT = "RECEIVER_ACCOUNT"
    TRANSACTION_AMOUNT = "TRANSACTION_AMOUNT"
    ORIGINAL_CURRENCY = "ORIGINAL_CURRENCY"
    USD_AMOUNT = "USD_AMOUNT"
    EUR_AMOUNT = "EUR_AMOUNT"
    SENDER_TYPE = "SENDER_TYPE"
    RECEIVER_TYPE = "RECEIVER_TYPE"
    SENDER_JURISDICTION = "SENDER_JURISDICTION"
    RECEIVER_JURISDICTION = "RECEIVER_JURISDICTION"
    SENDER_BANK_COUNTRY = "SENDER_BANK_COUNTRY"
    RECEIVER_BANK_COUNTRY = "RECEIVER_BANK_COUNTRY"
    PURPOSE = "PURPOSE"
    SOURCE_FILE = "SOURCE_FILE"
    INVESTIGATION = "INVESTIGATION"
    SENDER_CORE = "SENDER_CORE"
    RECEIVER_CORE = "RECEIVER_CORE"

class MappingStatus(str, enum.Enum):
    AUTO_ACCEPTED = "AUTO_ACCEPTED"
    NEEDS_CONFIRMATION = "NEEDS_CONFIRMATION"
    UNMAPPED = "UNMAPPED"
    CONFLICT = "CONFLICT"

@dataclass
class Candidate:
    canonical_concept: str
    score: float
    evidence: Dict[str, Any]

@dataclass
class ColumnMapping:
    source_column: str
    status: MappingStatus
    candidates: List[Candidate]
    selected_concept: Optional[str] = None
    mapper_version: str = "1.0"

@dataclass
class MappingManifest:
    mappings: List[ColumnMapping]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MappingManifest":
        maps = []
        for m in data.get("mappings", []):
            cands = [Candidate(**c) for c in m.get("candidates", [])]
            status = MappingStatus(m["status"]) if "status" in m else MappingStatus.UNMAPPED
            maps.append(ColumnMapping(
                source_column=m["source_column"],
                status=status,
                candidates=cands,
                selected_concept=m.get("selected_concept"),
                mapper_version=m.get("mapper_version", "1.0")
            ))
        return cls(mappings=maps)

def normalize_for_mapping(name: str) -> str:
    name = str(name).lower()
    name = name.replace('a/c', 'account')
    name = re.sub(r'[^a-z0-9]+', ' ', name)
    tokens = name.split()
    abbrevs = {
        'acct': 'account', 'no': 'number', 'num': 'number', 
        'txn': 'transaction', 'tx': 'transaction', 'amt': 'amount',
        'curr': 'currency', 'dt': 'date', 'ccy': 'currency', 'orig': 'original'
    }
    tokens = [abbrevs.get(t, t) for t in tokens]
    return '_'.join(tokens)

# Maintainable Semantic Alias Registry
SPECIFIC_ALIASES = {
    CanonicalConcept.TRANSACTION_ID: ["transaction_id", "transaction_number", "id", "ref", "reference"],
    CanonicalConcept.TIMESTAMP: ["timestamp", "date", "transaction_date", "created_at", "time",
                                 "datetime", "date_time", "transaction_datetime",
                                 "transaction_timestamp", "value_date", "booking_date",
                                 "posting_date", "execution_date"],
    CanonicalConcept.SENDER_NAME: ["sender_name", "payer_name", "originator_name", "from_name"],
    CanonicalConcept.RECEIVER_NAME: ["receiver_name", "beneficiary_name", "recipient_name", "to_name"],
    CanonicalConcept.SENDER_ACCOUNT: ["sender_account", "payer_account", "originator_account", "from_account", "sender_account_number"],
    CanonicalConcept.RECEIVER_ACCOUNT: ["receiver_account", "beneficiary_account", "recipient_account", "to_account", "receiver_account_number"],
    CanonicalConcept.TRANSACTION_AMOUNT: ["amount", "value", "transaction_amount", "original_amount", "amount_original"],
    CanonicalConcept.ORIGINAL_CURRENCY: ["currency", "original_currency", "transaction_currency",
                                         "payment_currency", "sent_currency"],
    CanonicalConcept.USD_AMOUNT: ["amount_usd", "usd_amount", "usd"],
    CanonicalConcept.EUR_AMOUNT: ["amount_eur", "eur_amount", "eur"],
    CanonicalConcept.SENDER_JURISDICTION: ["sender_jurisdiction", "sender_country"],
    CanonicalConcept.RECEIVER_JURISDICTION: ["receiver_jurisdiction", "receiver_country"],
    CanonicalConcept.PURPOSE: ["purpose", "description", "memo", "payment_type",
                               "transaction_type", "payment_method", "channel"],
    CanonicalConcept.INVESTIGATION: ["investigation", "is_laundering", "laundering_type", "label", "is_fraud", "fraud"]
}

GENERIC_ALIASES = {
    "account_number": [CanonicalConcept.SENDER_ACCOUNT, CanonicalConcept.RECEIVER_ACCOUNT],
    "account": [CanonicalConcept.SENDER_ACCOUNT, CanonicalConcept.RECEIVER_ACCOUNT],
    "customer_id": [CanonicalConcept.SENDER_ACCOUNT, CanonicalConcept.RECEIVER_ACCOUNT],
    "name": [CanonicalConcept.SENDER_NAME, CanonicalConcept.RECEIVER_NAME],
    "country": [CanonicalConcept.SENDER_JURISDICTION, CanonicalConcept.RECEIVER_JURISDICTION],
    "location": [CanonicalConcept.SENDER_JURISDICTION, CanonicalConcept.RECEIVER_JURISDICTION]
}

# ---------------------------------------------------------------------------
# Directional lexicon: role x concept-family
# ---------------------------------------------------------------------------
# The alias tables above enumerate whole phrases, so they only recognise a
# transaction side when the exact phrase was anticipated.  A column called
# ``Receiver_bank_location`` matches no specific alias, falls through to the
# direction-blind generic alias "location", and scores identically to
# ``Sender_bank_location`` -- so both resolve to a *sender* concept and collide.
#
# Direction is a property of the column name that is independent of what the
# column is about, so it is modelled as its own dimension.  A role token plus a
# concept-family token compose into a directional concept; adding "payee" to
# the role lexicon therefore teaches every receiver-side concept at once,
# rather than requiring one new phrase per concept per schema dialect.
#
# This affects only which concept is *proposed*.  Ambiguity still surfaces as
# NEEDS_CONFIRMATION for operator review, canonicalization is unchanged, and
# nothing here touches the frozen feature vocabulary or the checkpoint.

_SENDER_ROLE_TOKENS = frozenset({
    "sender", "sending", "sends", "payer", "payor", "originator", "originating",
    "remitter", "remitting", "debtor", "ordering", "from", "src",
    # Graph/ETL dialects express the same side as the edge tail.
    "source", "origin", "initiator", "outgoing",
})

_RECEIVER_ROLE_TOKENS = frozenset({
    "receiver", "receiving", "receives", "beneficiary", "benef", "recipient",
    "payee", "creditor", "destination", "dest", "to",
    # ... and the edge head.
    "target", "incoming",
})

#: Tokens that qualify a place as a *bank* location rather than a party's own
#: jurisdiction.  ``payer_bank_country`` feeds the model's 151-column country
#: block; ``payer_jurisdiction`` does not, so the two must not be conflated.
_BANK_TOKENS = frozenset({"bank", "banks", "branch", "institution", "swift", "bic"})

#: Tokens denoting a place, in either sense.
_PLACE_TOKENS = frozenset({
    "country", "countries", "jurisdiction", "location", "nation",
    "residence", "domicile", "iso",
})

_ACCOUNT_TOKENS = frozenset({"account", "iban", "wallet"})
#: Party words that become an identifier when paired with an id/number token.
_PARTY_TOKENS = frozenset({"customer", "client", "entity", "party", "counterparty"})
_ID_TOKENS = frozenset({"id", "identifier", "number", "no", "code", "ref"})
_NAME_TOKENS = frozenset({"name", "title"})
_TYPE_TOKENS = frozenset({"type", "category", "classification", "kind"})
_CORE_TOKENS = frozenset({"core"})

#: Tokens naming a point in time.  Modelled as a family for the same reason
#: direction is: ``event_time``, ``transaction_time``, ``settlement_date`` and
#: ``booking_datetime`` are one concept expressed in different dialects, and
#: enumerating whole phrases means every unanticipated dialect falls through to
#: a 0.1 substring match and never reaches confident selection.
_TEMPORAL_TOKENS = frozenset({
    "date", "time", "datetime", "timestamp", "day", "when",
})

#: Tokens naming the calendar half of an instant, and the clock half.  Used
#: only to decide whether two TIMESTAMP claimants are *competing* (two rival
#: readings of the same fact) or *complementary* (two halves of one fact).
_DATE_PART_TOKENS = frozenset({"date", "day"})
_TIME_PART_TOKENS = frozenset({"time", "clock", "hour"})

#: family -> (sender concept, receiver concept)
_DIRECTIONAL_FAMILIES = {
    "name": (CanonicalConcept.SENDER_NAME, CanonicalConcept.RECEIVER_NAME),
    "account": (CanonicalConcept.SENDER_ACCOUNT, CanonicalConcept.RECEIVER_ACCOUNT),
    "type": (CanonicalConcept.SENDER_TYPE, CanonicalConcept.RECEIVER_TYPE),
    "bank_country": (CanonicalConcept.SENDER_BANK_COUNTRY,
                     CanonicalConcept.RECEIVER_BANK_COUNTRY),
    "jurisdiction": (CanonicalConcept.SENDER_JURISDICTION,
                     CanonicalConcept.RECEIVER_JURISDICTION),
    "core": (CanonicalConcept.SENDER_CORE, CanonicalConcept.RECEIVER_CORE),
}

#: Families that must not be claimed without an explicit transaction side.
#: ``payment_type`` and ``transaction_type`` describe the payment, not the
#: parties, so the entity-type family only applies when a role or party token
#: says whose type it is.  ``core`` is a label whose bare token means nothing.
_ROLE_REQUIRED_FAMILIES = frozenset({"type", "core"})

#: Tokens marking a column as a *derived* form of another column rather than a
#: concept of its own: ``payer_name_norm`` beside ``payer_name``.  Both denote
#: the payer's name, so they tie and annihilate each other as a CONFLICT --
#: which is how the Reference Dataset's own raw file, whose schema carries both,
#: ends up with no counterparty identity at all.
#:
#: They are not rivals: the plain column is the fact, the qualified one is
#: somebody else's normalization of it.  The plain form therefore outranks it,
#: and the pipeline re-derives its own ``*_name_norm`` under the entity
#: resolution rule the checkpoint was trained with, rather than silently
#: adopting a foreign one.
_DERIVED_FORM_TOKENS = frozenset({
    "norm", "normalized", "normalised", "clean", "cleaned", "std",
    "standardized", "standardised", "canonical", "key",
})

#: Concepts describing what the payer **sent**.  The canonical schema has
#: ``amount_orig`` / ``amount_orig_currency`` and no counterpart for what the
#: beneficiary received, so a receiving-side column is a *different quantity*,
#: not another spelling of the same one.  Demoted rather than excluded: if the
#: receiving side is all a dataset has, it is still the best evidence available
#: and an operator can confirm it.
_SENT_SIDE_CONCEPTS = frozenset({
    CanonicalConcept.TRANSACTION_AMOUNT,
    CanonicalConcept.ORIGINAL_CURRENCY,
    CanonicalConcept.USD_AMOUNT,
    CanonicalConcept.EUR_AMOUNT,
})

#: Concept -> role, for the direction-contradiction penalty.
_CONCEPT_ROLE = {
    concept: role
    for sender_concept, receiver_concept in _DIRECTIONAL_FAMILIES.values()
    for concept, role in ((sender_concept, "sender"), (receiver_concept, "receiver"))
}


def detect_role(tokens: frozenset) -> Optional[str]:
    """Which side of the transaction a column name names, if either.

    ``None`` means the name is genuinely side-agnostic (``account_number``),
    which is ambiguity to be reviewed -- not an invitation to guess a side.
    """
    is_sender = bool(tokens & _SENDER_ROLE_TOKENS)
    is_receiver = bool(tokens & _RECEIVER_ROLE_TOKENS)
    if is_sender and not is_receiver:
        return "sender"
    if is_receiver and not is_sender:
        return "receiver"
    return None


def detect_family(tokens: frozenset) -> Optional[str]:
    """Which concept family a column name is about, if any.

    A pure function of the column's tokens: whether the family may then be
    *claimed* without a stated side is a separate rule, held once in
    ``_ROLE_REQUIRED_FAMILIES``.

    Evaluated most-specific first: ``bank_country`` outranks ``jurisdiction``
    because ``sender_bank_location`` carries both a bank token and a place
    token, and only the bank reading feeds the model's country block.
    """
    has_place = bool(tokens & _PLACE_TOKENS)
    has_bank = bool(tokens & _BANK_TOKENS)
    if has_place and has_bank:
        return "bank_country"
    if has_place:
        return "jurisdiction"
    if has_bank:
        # A bank token with no place token names the *institution*, not the
        # party and not a country.  IBM AML's `payer_bank_account` holds 78,394
        # distinct bank identifiers; reading it as the payer's own account
        # would silently replace the entity population with banks.  Nothing is
        # claimed rather than claiming the wrong thing.
        return None
    if tokens & _ACCOUNT_TOKENS:
        return "account"
    if tokens & _PARTY_TOKENS and tokens & _ID_TOKENS:
        return "account"
    if detect_role(tokens) is not None and tokens & _ID_TOKENS:
        # `originator_id`, `beneficiary_id`: a role token *is* a party token
        # once an id token says the column identifies that party.  Without
        # this, "id" matched TRANSACTION_ID by substring and the two entity
        # columns of a whole schema dialect were read as the transaction's own
        # identifier.
        return "account"
    if tokens & _NAME_TOKENS:
        return "name"
    if tokens & _TYPE_TOKENS:
        return "type"
    if tokens & _CORE_TOKENS:
        return "core"
    return None


def names_only_a_party(tokens: frozenset) -> bool:
    """Whether a column name consists of nothing but a transaction-side word.

    ``payer`` / ``payee`` / ``sender`` / ``from`` name a party and say nothing
    about *which* attribute of that party the column holds.  ``amount_dest``
    also contains a role token, but it is qualified by a token that names a
    different concept entirely, so it is not a party column.
    """
    roles = _SENDER_ROLE_TOKENS | _RECEIVER_ROLE_TOKENS
    meaningful = {t for t in tokens if t}
    return bool(meaningful) and meaningful <= roles


def compositional_lexical_score(
    tokens: frozenset, concept: CanonicalConcept
) -> Optional[float]:
    """Score ``concept`` from a role token composed with a family token.

    Returns ``None`` when the composition says nothing about this concept, so
    the caller can fall back to the phrase-based alias tables.

    * role present and matching        -> 0.6, as strong as an exact alias
    * role absent                      -> 0.3, ambiguous across both sides
    * role present and contradicting   -> negative, so a column that names one
      side is never proposed as the other
    """
    role = detect_role(tokens)
    family = detect_family(tokens)

    concept_role = _CONCEPT_ROLE.get(concept)
    if concept_role is not None and role is not None and concept_role != role:
        # The column names a side, and it is not this concept's side.  This is
        # evidence *against*, not merely absence of evidence for.
        return -0.4

    if concept == CanonicalConcept.TRANSACTION_ID and role is not None:
        # A transaction id identifies the *transaction*.  A column that names a
        # party is identifying that party, whatever else its name contains.
        return -0.4

    if family is None:
        if role is not None and names_only_a_party(tokens):
            # An unqualified party word.  Which attribute of the party it holds
            # is not stated, so both identity readings are proposed and the
            # type/cardinality evidence below discriminates: the unqualified
            # reading of a party word is its label (a name), while an integer
            # or identifier-shaped column is an account.  Neither is asserted
            # here -- this only puts both on the table for review.
            sender_name, receiver_name = _DIRECTIONAL_FAMILIES["name"]
            sender_acct, receiver_acct = _DIRECTIONAL_FAMILIES["account"]
            if concept in (sender_name, receiver_name):
                return 0.45
            if concept in (sender_acct, receiver_acct):
                return 0.3
        return None
    if family in _ROLE_REQUIRED_FAMILIES and role is None:
        return None

    sender_concept, receiver_concept = _DIRECTIONAL_FAMILIES[family]
    if concept not in (sender_concept, receiver_concept):
        return None

    if role is None:
        # Side-agnostic: both directions are equally supported, which the
        # caller surfaces as NEEDS_CONFIRMATION rather than a guess.
        return 0.3
    return 0.6


def temporal_lexical_score(tokens: frozenset) -> Optional[float]:
    """Score TIMESTAMP from a temporal token, independent of the surrounding phrase.

    Returns ``None`` when the name says nothing temporal.  Time is not a
    directional concept, so unlike :func:`compositional_lexical_score` there is
    no role dimension: any token naming a point in time supports TIMESTAMP as
    strongly as an exact alias would.
    """
    if tokens & _TEMPORAL_TOKENS:
        return 0.6
    return None


def _names_a_complete_instant(norm: str, tokens: frozenset) -> bool:
    """Whether a column name claims a whole instant rather than half of one.

    ``transaction_datetime`` and ``date_time`` name a full instant; ``Date``
    and ``Time`` each name half of one.  Used only to rank TIMESTAMP claimants
    against each other.
    """
    return ("datetime" in norm or "timestamp" in norm
            or {"date", "time"} <= tokens)


def timestamp_part(column_name: str) -> Optional[str]:
    """Which half of an instant a column name claims: ``"date"``, ``"time"``, or None.

    ``None`` covers both "not temporal at all" and "a whole instant" — in
    either case the column is not one half of a pair, which is the only thing
    callers use this to decide.  Semantic, not positional: it reads the same
    token vocabulary the rest of the mapper does, so no caller needs to know
    that some particular dataset happens to call its columns ``Date`` and
    ``Time``.
    """
    norm = normalize_for_mapping(column_name)
    tokens = frozenset(norm.split('_'))
    if _names_a_complete_instant(norm, tokens):
        return None
    if tokens & _DATE_PART_TOKENS:
        return "date"
    if tokens & _TIME_PART_TOKENS:
        return "time"
    return None


def get_concept_expected_type(concept: CanonicalConcept) -> str:
    if concept == CanonicalConcept.TIMESTAMP:
        return "datetime"
    if concept in (CanonicalConcept.TRANSACTION_AMOUNT, CanonicalConcept.USD_AMOUNT, CanonicalConcept.EUR_AMOUNT):
        return "float" # or integer
    return "string"

def score_column_for_concept(col: ColumnProfile, concept: CanonicalConcept) -> Candidate:
    norm = normalize_for_mapping(col.original_name)
    tokens = frozenset(norm.split('_'))
    lexical_score = 0.0

    # 1. Lexical Matching
    specifics = SPECIFIC_ALIASES.get(concept, [])
    if norm in specifics:
        lexical_score = 0.6
    else:
        # Check generic alias
        for gen_alias, concepts in GENERIC_ALIASES.items():
            if norm == gen_alias and concept in concepts:
                lexical_score = 0.3
                break
        if lexical_score == 0.0:
            # Partial match
            for sp in specifics:
                if sp in norm or norm in sp:
                    lexical_score = 0.1
                    break
            if lexical_score == 0.0:
                for gen_alias in GENERIC_ALIASES:
                    if gen_alias in norm and concept in GENERIC_ALIASES[gen_alias]:
                        lexical_score = 0.1
                        break

    # 1b. Directional composition (role x family).  Takes the stronger reading
    # when it agrees, and overrides outright when it *contradicts*: a column
    # naming one side of the transaction must never be proposed as the other,
    # however well the phrase tables happen to match it.
    composed = compositional_lexical_score(tokens, concept)
    if composed is not None:
        if composed < 0:
            lexical_score = composed
        else:
            lexical_score = max(lexical_score, composed)

    # 1b-ii. Temporal composition, on the same footing: a token naming a point
    # in time supports TIMESTAMP however the phrase around it is spelled.
    if concept == CanonicalConcept.TIMESTAMP and lexical_score >= 0:
        temporal = temporal_lexical_score(tokens)
        if temporal is not None:
            lexical_score = max(lexical_score, temporal)

    # 1c. Timestamp completeness.  A column supplying a whole instant is
    # strictly better evidence for TIMESTAMP than one supplying only the date
    # half or only the time half, so a `transaction_datetime` outranks a
    # `Date` / `Time` pair instead of tying with them and being discarded as a
    # three-way conflict.  A general property of the concept, not of any
    # particular dataset.
    if (concept == CanonicalConcept.TIMESTAMP and lexical_score > 0
            and _names_a_complete_instant(norm, tokens)):
        lexical_score += 0.1

    # 1d. Demotions.  Neither disqualifies a column; both break a tie that
    # would otherwise be resolved by discarding both claimants.
    if lexical_score > 0 and tokens & _DERIVED_FORM_TOKENS:
        # A normalization of the fact ranks below the fact.
        lexical_score *= 0.5
    if (lexical_score > 0 and concept in _SENT_SIDE_CONCEPTS
            and detect_role(tokens) == "receiver"):
        # `amount_dest` is what arrived, not what was sent.
        lexical_score *= 0.5

    # 2. Type Compatibility
    type_score = 0.0
    expected = get_concept_expected_type(concept)
    if expected == "float" and col.inferred_type in ("float", "integer"):
        type_score = 0.2 if lexical_score > 0 else 0.05
    elif expected == col.inferred_type:
        type_score = 0.2 if lexical_score > 0 else 0.05
    elif col.inferred_type == "mixed/unknown":
        type_score = 0.05
    elif expected == "string" and col.inferred_type == "integer":
        # Account numbers might be inferred as integers
        if concept in (CanonicalConcept.SENDER_ACCOUNT, CanonicalConcept.RECEIVER_ACCOUNT, CanonicalConcept.TRANSACTION_ID):
            type_score = 0.1
        elif concept in (CanonicalConcept.SENDER_NAME, CanonicalConcept.RECEIVER_NAME):
            # A counterparty name is a label, not a number.  A numeric column
            # under a party word is that party's identifier.
            type_score = -0.3
    else:
        # Penalty for blatant mismatch (e.g. amount is datetime)
        type_score = -0.3

    # 3. Cardinality & Pattern
    cardinality_score = 0.0
    pattern_score = 0.0
    
    if concept == CanonicalConcept.TRANSACTION_ID:
        if col.uniqueness_ratio > 0.9:
            cardinality_score = 0.2
        if "identifier-like" in col.pattern_hints:
            pattern_score = 0.1
    elif concept in (CanonicalConcept.SENDER_ACCOUNT, CanonicalConcept.RECEIVER_ACCOUNT):
        # Accounts should have some repetition in graph, but also many uniques.
        if 0.05 < col.uniqueness_ratio < 0.9:
            cardinality_score = 0.1
        if col.uniqueness_ratio < 0.01:
            # suspiciously low uniqueness for an account-like column
            cardinality_score = -0.2
    elif concept in (CanonicalConcept.SENDER_JURISDICTION, CanonicalConcept.RECEIVER_JURISDICTION,
                     CanonicalConcept.SENDER_BANK_COUNTRY, CanonicalConcept.RECEIVER_BANK_COUNTRY):
        # Countries and jurisdictions are a small closed set repeated across
        # many rows.  A "bank country" column with near-unique values is a bank
        # account identifier, not a country -- exactly the IBM AML case.
        if col.uniqueness_ratio < 0.05:
            cardinality_score = 0.2
        elif col.uniqueness_ratio > 0.5:
            cardinality_score = -0.3
    elif concept in (CanonicalConcept.SENDER_TYPE, CanonicalConcept.RECEIVER_TYPE):
        # An entity type is a handful of categories (Company / Person / ...).
        if col.uniqueness_ratio < 0.05:
            cardinality_score = 0.2
        elif col.uniqueness_ratio > 0.5:
            cardinality_score = -0.3
    if (concept in (CanonicalConcept.SENDER_NAME, CanonicalConcept.RECEIVER_NAME)
            and "identifier-like" in col.pattern_hints):
        # A near-unique column is shaped like an identifier, not like a
        # population of counterparty labels.  This is what separates the two
        # readings of an unqualified party column (`payer`, `payee`) when its
        # values are strings: evidence about the values, not about the spelling.
        pattern_score = -0.1


    final_score = lexical_score + type_score + cardinality_score + pattern_score
    
    # Clip to [0, 1.0] if >0, allow negative if penalized heavily
    if final_score > 1.0: final_score = 1.0
    
    evidence = {
        "lexical_score": lexical_score,
        "type_score": type_score,
        "cardinality_score": cardinality_score,
        "pattern_score": pattern_score,
        "final_score": round(final_score, 3)
    }
    return Candidate(canonical_concept=concept.value, score=round(final_score, 3), evidence=evidence)

def resolve_mappings(profile: DatasetProfile) -> MappingManifest:
    mappings = []
    
    # First pass: generate candidates for each column
    column_candidates = {}
    for col in profile.columns:
        cands = []
        for concept in CanonicalConcept:
            cand = score_column_for_concept(col, concept)
            # Type and cardinality are corroborating evidence, not identifying
            # evidence: without any lexical signal they describe the shape of a
            # column, never what it means.  Requiring lexical support stops a
            # low-cardinality string column such as `source_file` from being
            # offered as an entity type purely because it repeats.
            if cand.score >= 0.2 and cand.evidence["lexical_score"] > 0:
                cands.append(cand)
        cands.sort(key=lambda x: x.score, reverse=True)
        column_candidates[col.original_name] = cands

    # Second pass: per-column state assignment, before cross-column arbitration.
    for col_name, cands in column_candidates.items():
        if not cands:
            mappings.append(ColumnMapping(col_name, MappingStatus.UNMAPPED, []))
            continue

        top_cand = cands[0]
        status = MappingStatus.UNMAPPED
        selected = None

        # A tie between two concepts *within one column* is the same kind of
        # ambiguity as a tie between two columns, and must be resolved the same
        # way: not at all.  Previously the sort was stable, so an exact tie was
        # broken by CanonicalConcept declaration order -- a bare `country`
        # column silently became SENDER_JURISDICTION and a bare
        # `account_number` silently became SENDER_ACCOUNT, choosing a side of
        # the transaction on the strength of an enum's line ordering.  The
        # column is still surfaced with all its candidates for review.
        tied_at_top = [c for c in cands if c.score == top_cand.score]

        if len(tied_at_top) > 1:
            status = (MappingStatus.NEEDS_CONFIRMATION if top_cand.score >= 0.3
                      else MappingStatus.UNMAPPED)
        elif top_cand.score >= 0.8:
            status = MappingStatus.AUTO_ACCEPTED
            selected = top_cand.canonical_concept
        elif top_cand.score >= 0.3:
            status = MappingStatus.NEEDS_CONFIRMATION
            selected = top_cand.canonical_concept

        mappings.append(ColumnMapping(
            source_column=col_name,
            status=status,
            candidates=cands,
            selected_concept=selected
        ))

    # Third pass: cross-column arbitration.
    #
    # Two columns claiming one concept are only genuinely in CONFLICT when the
    # evidence does not distinguish them.  Previously the *first* claimant was
    # registered and every later one was rejected regardless of score, so a
    # strongly-evidenced column lost to a weak one that merely came first, and
    # both were discarded -- a `Date` + `Time` pair alongside a real datetime
    # column annihilated the timestamp concept entirely.
    #
    # Rank claimants by score instead.  A clear winner is kept; the columns it
    # beat drop to NEEDS_CONFIRMATION so an operator still sees them and can
    # override.  A tie is still a real conflict and is still reported as one.
    by_concept: Dict[str, List[ColumnMapping]] = {}
    for m in mappings:
        if m.selected_concept:
            by_concept.setdefault(m.selected_concept, []).append(m)

    def _score_for(mapping: ColumnMapping, concept: str) -> float:
        for c in mapping.candidates:
            if c.canonical_concept == concept:
                return c.score
        return 0.0

    def _are_complementary_instant_halves(claimants: List[ColumnMapping]) -> bool:
        """Exactly one calendar column and one clock column, and nothing else."""
        if len(claimants) != 2:
            return False
        parts = sorted(timestamp_part(m.source_column) for m in claimants
                       if timestamp_part(m.source_column) is not None)
        return parts == ["date", "time"]

    for concept, claimants in by_concept.items():
        if len(claimants) < 2:
            continue

        if (concept == CanonicalConcept.TIMESTAMP.value
                and _are_complementary_instant_halves(claimants)):
            # Not rival readings of one fact: two halves of one fact.  A
            # calendar column and a clock column each name only half an
            # instant, and together name a whole one, so neither beats the
            # other and neither is a conflict.  Both keep the concept and
            # canonicalization composes them (src/datasets/derivation.py).
            # NEEDS_CONFIRMATION because a composed field is a derivation the
            # operator should see, not an unremarkable direct mapping.
            for m in claimants:
                m.status = MappingStatus.NEEDS_CONFIRMATION
                m.selected_concept = concept
            continue

        claimants.sort(key=lambda m: _score_for(m, concept), reverse=True)
        best_score = _score_for(claimants[0], concept)
        tied = [m for m in claimants if _score_for(m, concept) == best_score]

        if len(tied) > 1:
            # Indistinguishable evidence: neither may be auto-selected.
            for m in tied:
                m.status = MappingStatus.CONFLICT
                m.selected_concept = None
            losers = claimants[len(tied):]
        else:
            losers = claimants[1:]

        for m in losers:
            # Beaten, not invalid.  The selection is cleared -- two columns
            # holding one concept would both be projected into the same
            # canonical field and concatenated -- but the candidate list is
            # kept, so the column stays visible for operator review instead of
            # being silently dropped.
            m.status = MappingStatus.NEEDS_CONFIRMATION
            m.selected_concept = None

    return MappingManifest(mappings=mappings)
