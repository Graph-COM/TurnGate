import json
import random
import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# fix all random seeds for reproducibility
random.seed(42)
# numpy and torch
from numpy.random import seed as np_seed

np_seed(42)

load_dotenv()

try:
    import numpy as np
    from google import genai
    from google.genai import types
    from sklearn.metrics.pairwise import cosine_similarity

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print(
        "Warning: numpy, google-genai, or scikit-learn not found. Embedding features will be disabled."
    )


class Turn:
    def __init__(
        self,
        role: str,
        content: str,
        is_harmful: bool = False,
        original_info: Dict = None,
    ):
        self.role = role
        self.content = content
        self.is_harmful = is_harmful
        self.original_info = original_info or {}

    def to_dict(self):
        return {
            "role": self.role,
            "content": self.content,
            "is_harmful": self.is_harmful,
            "original_info": self.original_info,
        }

    def __repr__(self):
        return f"Turn(role={self.role}, is_harmful={self.is_harmful})"


class Conversation:
    def __init__(
        self, turns: List[Turn], metadata: Dict = None, conv_type: str = "unknown"
    ):
        self.turns = turns
        self.metadata = metadata or {}
        self.conv_type = conv_type  # 'harmful', 'benign', 'mixed'

    def get_full_text(self) -> str:
        """Concatenates all turns into a single string for embedding."""
        return "\n".join([f"{t.role}: {t.content}" for t in self.turns])

    @property
    def last_harmful_turn_index(self) -> int:
        """Returns the index of the last harmful user turn."""
        last_idx = -1
        for i, turn in enumerate(self.turns):
            if turn.is_harmful and turn.role == "user":
                last_idx = i
        return last_idx

    def get_history(self, up_to_index: int) -> List[Dict]:
        """Returns the conversation history up to (but not including) the given index."""
        return [t.to_dict() for t in self.turns[:up_to_index]]


class ConversationMixer:
    def __init__(self, benign_path: str, harmful_path: str):
        self.benign_path = benign_path
        self.harmful_path = harmful_path
        self.benign_data = self._load_jsonl(benign_path)
        self.harmful_data = self._load_jsonl(harmful_path)
        self._client = None

    def _get_client(self):
        if self._client is None:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable is not set.")
            self._client = genai.Client(api_key=api_key)
        return self._client

    def _load_jsonl(self, path: str) -> List[Dict]:
        data = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        except FileNotFoundError:
            print(f"Warning: File not found: {path}")
        return data

    def get_benign_conversations(self, min_turns: int = 0) -> List[Conversation]:
        """Load benign conversations. All turns marked is_harmful=False."""
        return self._load_conversations(
            self.benign_data, is_harmful=False, conv_type="benign", min_turns=min_turns
        )

    def get_harmful_conversations(self, min_turns: int = 0) -> List[Conversation]:
        """Load harmful conversations. All turns marked is_harmful=True. Last turn is the target to block."""
        return self._load_conversations(
            self.harmful_data, is_harmful=True, conv_type="harmful", min_turns=min_turns
        )

    def _load_conversations(
        self, data: List[Dict], is_harmful: bool, conv_type: str, min_turns: int = 0
    ) -> List[Conversation]:
        """Helper to load and convert raw JSONL data to Conversation objects."""
        convs = []
        for item in data:
            turns_data = item.get("conversation", [])
            if len(turns_data) >= min_turns:
                turns = [
                    Turn(
                        t["role"], t["content"], is_harmful=is_harmful, original_info=t
                    )
                    for t in turns_data
                ]
                convs.append(Conversation(turns, metadata=item, conv_type=conv_type))
        return convs

    def _get_embeddings(
        self, texts: List[str], model_name: str = "gemini-embedding-001"
    ) -> Any:
        """Helper to get embeddings using Gemini API."""
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "Packages 'google-genai' and 'numpy' are required for embeddings."
            )

        client = self._get_client()

        all_embeddings = []
        batch_size = 100  # API limit is 100

        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                # google-genai Client.models.embed_content
                result = client.models.embed_content(
                    model=model_name,
                    contents=batch,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
                )
                # The result object has an 'embeddings' attribute which is a list of ContentEmbedding objects
                batch_embeddings = [e.values for e in result.embeddings]
                all_embeddings.extend(batch_embeddings)

            return np.array(all_embeddings)
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            # Return zero vector fallback
            return np.zeros((len(texts), 768))

    def save_embeddings(self, embeddings: np.ndarray, path: str):
        """Saves embeddings to a .npy file."""
        try:
            np.save(path, embeddings)
            print(f"Embeddings saved to {path}")
        except Exception as e:
            print(f"Error saving embeddings: {e}")

    def load_embeddings(self, path: str) -> Optional[np.ndarray]:
        """Loads embeddings from a .npy file."""
        try:
            if os.path.exists(path):
                print(f"Loading embeddings from {path}...")
                return np.load(path)
        except Exception as e:
            print(f"Error loading embeddings: {e}")
        return None

    def _get_ua_pairs(self, conv: Conversation) -> List[tuple]:
        """Extract all user-assistant pairs from a conversation."""
        pairs = []
        for i in range(len(conv.turns) - 1):
            if conv.turns[i].role == "user" and conv.turns[i + 1].role == "assistant":
                pairs.append((conv.turns[i], conv.turns[i + 1]))
        return pairs

    def _select_benign_conversation(
        self,
        benign_pool: List[Conversation],
        benign_embeddings: Any,
        context_embedding: Any,
        strategy: str,
    ) -> Conversation:
        """Select a benign conversation based on strategy (random or similarity)."""
        if (
            strategy == "similarity"
            and context_embedding is not None
            and benign_embeddings is not None
        ):
            sims = cosine_similarity([context_embedding], benign_embeddings)[0]
            top_indices = sims.argsort()[-5:][::-1]  # Top 5 for variety
            return benign_pool[random.choice(top_indices)]
        return random.choice(benign_pool)

    def _get_num_insertions(
        self, total_len: int, ratio: float, min_val: int, max_val: int
    ) -> int:
        """Calculate number of insertions based on ratio or range."""
        if ratio is not None:
            return int(total_len * ratio)
        return random.randint(min_val, max_val)

    def _get_context_embedding(
        self, conv: Conversation, strategy: str, pool_embeddings: Any
    ) -> Optional[Any]:
        """Get context embedding for a conversation if similarity strategy is used."""
        if (
            strategy == "similarity"
            and GEMINI_AVAILABLE
            and pool_embeddings is not None
            and conv.turns
        ):
            return self._get_embeddings([conv.get_full_text()])[0]
        return None

    def _prepare_pool_embeddings(
        self,
        pool: List[Conversation],
        strategy: str,
        pool_type: str,
        cached_path: str = None,
    ) -> Optional[Any]:
        """Load or compute embeddings for a conversation pool."""
        if strategy != "similarity" or not GEMINI_AVAILABLE:
            return None

        if cached_path:
            embeddings = self.load_embeddings(cached_path)
            if embeddings is not None:
                return embeddings

        print(f"Generating embeddings for {pool_type} pool...")
        texts = [c.get_full_text() for c in pool]
        embeddings = self._get_embeddings(texts)

        if cached_path:
            self.save_embeddings(embeddings, cached_path)

        return embeddings

    def generate_mixed_benign(
        self,
        insert_ratio: float = None,
        min_insert: int = 0,
        max_insert: int = 2,
        strategy: str = "random",  # 'random' or 'similarity'
        min_harmful_conv_turns: int = 4,
        cached_harmful_embeddings_path: str = None,
    ) -> List[Conversation]:
        """
        Generates BENIGN mixed conversations by inserting harmful turns from DIFFERENT conversations into benign ones.
        """
        harmful_pool = self.get_harmful_conversations(min_turns=min_harmful_conv_turns)
        benign_pool = self.get_benign_conversations(min_turns=2)

        if not benign_pool or not harmful_pool:
            print(
                f"Insufficient data: benign={len(benign_pool)}, harmful={len(harmful_pool)}"
            )
            return []

        harmful_embeddings = self._prepare_pool_embeddings(
            harmful_pool, strategy, "harmful", cached_harmful_embeddings_path
        )

        # Pre-calculate all harmful pairs and their embeddings if using similarity
        all_harmful_pairs = []
        for h_conv in harmful_pool:
            all_harmful_pairs.extend(self._get_ua_pairs(h_conv))

        harmful_turn_embeddings = None
        if (
            strategy == "similarity"
            and harmful_embeddings is not None
            and all_harmful_pairs
        ):
            print("Pre-generating turn-level embeddings for harmful pool...")
            harmful_turn_embeddings = self._get_embeddings(
                [p[0].content for p in all_harmful_pairs]
            )

        mixed_convs = []
        for b_conv in benign_pool:
            num_to_insert = self._get_num_insertions(
                len(b_conv.turns), insert_ratio, min_insert, max_insert
            )
            if num_to_insert == 0:
                continue

            b_context_embedding = self._get_context_embedding(
                b_conv, strategy, harmful_embeddings
            )

            # Select harmful turns
            if (
                strategy == "similarity"
                and b_context_embedding is not None
                and harmful_turn_embeddings is not None
            ):
                sims = cosine_similarity(
                    [b_context_embedding], harmful_turn_embeddings
                )[0]
                top_indices = sims.argsort()[-num_to_insert * 2 :][::-1]
                selected_indices = random.sample(
                    list(top_indices[: min(len(top_indices), num_to_insert * 2)]),
                    min(num_to_insert, len(top_indices)),
                )
                selected_turns = [all_harmful_pairs[i] for i in selected_indices]
            else:
                selected_turns = random.sample(
                    all_harmful_pairs, min(num_to_insert, len(all_harmful_pairs))
                )

            if not selected_turns:
                continue

            new_turns = list(b_conv.turns)
            possible_positions = {0, len(new_turns)}
            possible_positions.update(
                [i + 1 for i, t in enumerate(new_turns) if t.role == "assistant"]
            )
            possible_positions = list(possible_positions)

            insert_positions = sorted(
                random.sample(
                    possible_positions
                    * ((num_to_insert // len(possible_positions)) + 1),
                    len(selected_turns),
                ),
                reverse=True,
            )

            for pos, (h_user, h_assistant) in zip(insert_positions, selected_turns):
                new_turns.insert(
                    pos, Turn(h_user.role, h_user.content, False, h_user.original_info)
                )
                new_turns.insert(
                    pos + 1,
                    Turn(
                        h_assistant.role,
                        h_assistant.content,
                        False,
                        h_assistant.original_info,
                    ),
                )

            mixed_convs.append(
                Conversation(
                    new_turns,
                    metadata={
                        "original_benign_id": b_conv.metadata.get("sample_index"),
                        "is_mixed_benign": True,
                        "strategy": strategy,
                        "num_insertions": len(selected_turns),
                    },
                    conv_type="benign",
                )
            )

        return mixed_convs

    def generate_mixed_harmful(
        self,
        insert_ratio: float = None,
        min_insert_per_turn: int = 0,
        max_insert_per_turn: int = 2,
        strategy: str = "random",  # 'random' or 'similarity'
        min_turns_filter: int = 4,
        cached_benign_embeddings_path: str = None,
    ) -> List[Conversation]:
        """
        Generates HARMFUL mixed conversations by inserting related benign turns BETWEEN harmful turns.
        """
        harmful_pool = self.get_harmful_conversations(min_turns=min_turns_filter)
        benign_pool = self.get_benign_conversations(min_turns=min_turns_filter)

        if not benign_pool:
            return harmful_pool

        benign_embeddings = self._prepare_pool_embeddings(
            benign_pool, strategy, "benign", cached_benign_embeddings_path
        )

        mixed_convs = []
        for h_conv in harmful_pool:
            mixed_turns = []
            h_context_embedding = self._get_context_embedding(
                h_conv, strategy, benign_embeddings
            )

            total_pairs_to_insert = (
                self._get_num_insertions(
                    sum(1 for t in h_conv.turns if t.role == "user"), insert_ratio, 0, 0
                )
                if insert_ratio
                else None
            )
            pairs_inserted = 0

            for i, turn in enumerate(h_conv.turns + [None]):
                if turn is None or turn.role == "user":
                    if insert_ratio is not None:
                        remaining_points = (
                            sum(1 for t in h_conv.turns[i:] if t.role == "user") + 1
                        )
                        remaining_pairs = total_pairs_to_insert - pairs_inserted
                        num_pairs = (
                            random.randint(
                                0,
                                min(
                                    remaining_pairs,
                                    max(0, remaining_pairs // remaining_points + 1),
                                ),
                            )
                            if remaining_points > 0
                            else 0
                        )
                    else:
                        num_pairs = random.randint(
                            min_insert_per_turn, max_insert_per_turn
                        )

                    if num_pairs > 0:
                        b_conv = self._select_benign_conversation(
                            benign_pool,
                            benign_embeddings,
                            h_context_embedding,
                            strategy,
                        )
                        b_pairs = self._get_ua_pairs(b_conv)
                        if b_pairs:
                            actual_pairs = min(num_pairs, len(b_pairs))
                            start_idx = random.randint(0, len(b_pairs) - actual_pairs)
                            mixed_turns.extend(
                                [
                                    Turn(t.role, t.content, False, t.original_info)
                                    for pair in b_pairs[
                                        start_idx : start_idx + actual_pairs
                                    ]
                                    for t in pair
                                ]
                            )
                            pairs_inserted += actual_pairs

                if turn is not None:
                    mixed_turns.append(turn)

            mixed_convs.append(
                Conversation(
                    mixed_turns,
                    metadata={
                        "original_harmful_id": h_conv.metadata.get("sample_index"),
                        "is_mixed_harmful": True,
                        "strategy": strategy,
                    },
                    conv_type="mixed",
                )
            )

        return mixed_convs


class RolloutDataLoader:
    """
    DataLoader for rollout data where each sample has multiple rollouts.
    Supports loading benign and harmful rollouts with fixed train/valid/test split (70/15/15).
    """

    def __init__(
        self,
        benign_path: str,
        harmful_path: str,
        rollouts_per_sample: int = 20,
        train_ratio: float = 0.7,
        valid_ratio: float = 0.15,
        seed: int = 42,
    ):
        """
        Initialize the RolloutDataLoader.

        Args:
            benign_path: Path to benign rollout JSONL file
            harmful_path: Path to harmful rollout JSONL file
            rollouts_per_sample: Number of rollouts per sample (default: 20)
            train_ratio: Training set ratio (default: 0.7)
            valid_ratio: Validation set ratio (default: 0.15)
            seed: Random seed for reproducibility (default: 42)
        """
        self.benign_path = benign_path
        self.harmful_path = harmful_path
        self.rollouts_per_sample = rollouts_per_sample
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.seed = seed

        random.seed(seed)
        np_seed(seed)

        # Load raw data
        self.benign_rollouts = self._load_jsonl(benign_path)
        self.harmful_rollouts = self._load_jsonl(harmful_path)

        # Group rollouts by sample_index
        self.benign_samples = self._group_by_sample(self.benign_rollouts)
        self.harmful_samples = self._group_by_sample(self.harmful_rollouts)

        # Create train/valid/test split
        self.benign_train, self.benign_valid, self.benign_test = self._split_samples(
            self.benign_samples
        )
        self.harmful_train, self.harmful_valid, self.harmful_test = self._split_samples(
            self.harmful_samples
        )

    @staticmethod
    def _load_jsonl(path: str) -> List[Dict]:
        """Load JSONL file."""
        data = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        except FileNotFoundError:
            print(f"Warning: File not found: {path}")
        return data

    @staticmethod
    def _group_by_sample(rollouts: List[Dict]) -> Dict[int, List[Dict]]:
        """
        Group rollouts by sample_index.

        Args:
            rollouts: List of rollout dictionaries

        Returns:
            Dictionary mapping sample_index to list of rollouts
        """
        samples = {}
        for rollout in rollouts:
            sample_idx = rollout.get("sample_index", 0)
            if sample_idx not in samples:
                samples[sample_idx] = []
            samples[sample_idx].append(rollout)
        return samples

    def _split_samples(
        self, samples: Dict[int, List[Dict]]
    ) -> tuple[Dict[int, List[Dict]], Dict[int, List[Dict]], Dict[int, List[Dict]]]:
        """
        Split samples into train, valid, and test sets with fixed ratios.

        Args:
            samples: Dictionary of samples

        Returns:
            Tuple of (train_samples, valid_samples, test_samples)
        """
        sample_indices = sorted(samples.keys())
        train_split = int(len(sample_indices) * self.train_ratio)
        valid_split = int(len(sample_indices) * (self.train_ratio + self.valid_ratio))

        train_indices = sample_indices[:train_split]
        valid_indices = sample_indices[train_split:valid_split]
        test_indices = sample_indices[valid_split:]

        train_samples = {idx: samples[idx] for idx in train_indices}
        valid_samples = {idx: samples[idx] for idx in valid_indices}
        test_samples = {idx: samples[idx] for idx in test_indices}

        return train_samples, valid_samples, test_samples

    def get_train_data(self, conv_type: str = "both") -> List[Conversation]:
        """
        Get training data as Conversation objects.

        Args:
            conv_type: Type of conversations to retrieve ('benign', 'harmful', or 'both')

        Returns:
            List of Conversation objects
        """
        return self._samples_to_conversations(
            self.benign_train, self.harmful_train, conv_type
        )

    def get_valid_data(self, conv_type: str = "both") -> List[Conversation]:
        """
        Get validation data as Conversation objects.

        Args:
            conv_type: Type of conversations to retrieve ('benign', 'harmful', or 'both')

        Returns:
            List of Conversation objects
        """
        return self._samples_to_conversations(
            self.benign_valid, self.harmful_valid, conv_type
        )

    def get_test_data(self, conv_type: str = "both") -> List[Conversation]:
        """
        Get test data as Conversation objects.

        Args:
            conv_type: Type of conversations to retrieve ('benign', 'harmful', or 'both')

        Returns:
            List of Conversation objects
        """
        return self._samples_to_conversations(
            self.benign_test, self.harmful_test, conv_type
        )

    def _samples_to_conversations(
        self,
        benign_samples: Dict[int, List[Dict]],
        harmful_samples: Dict[int, List[Dict]],
        conv_type: str = "both",
    ) -> List[Conversation]:
        """
        Convert sample dictionaries to Conversation objects.

        Args:
            benign_samples: Dictionary of benign samples
            harmful_samples: Dictionary of harmful samples
            conv_type: Type to retrieve ('benign', 'harmful', or 'both')

        Returns:
            List of Conversation objects
        """
        conversations = []

        if conv_type in ["benign", "both"]:
            for sample_idx, rollouts in benign_samples.items():
                for rollout in rollouts:
                    conv = self._rollout_to_conversation(rollout, "benign")
                    if conv:
                        conversations.append(conv)

        if conv_type in ["harmful", "both"]:
            for sample_idx, rollouts in harmful_samples.items():
                for rollout in rollouts:
                    conv = self._rollout_to_conversation(rollout, "harmful")
                    if conv:
                        conversations.append(conv)

        return conversations

    @staticmethod
    def _rollout_to_conversation(
        rollout: Dict, conv_type: str
    ) -> Optional[Conversation]:
        """
        Convert a single rollout to a Conversation object.

        Args:
            rollout: Single rollout dictionary
            conv_type: Type of conversation ('benign' or 'harmful')

        Returns:
            Conversation object or None if invalid
        """
        turns_data = rollout.get("conversation", [])
        if not turns_data:
            return None

        is_harmful = conv_type == "harmful"
        turns = [
            Turn(t["role"], t["content"], is_harmful=is_harmful, original_info=t)
            for t in turns_data
        ]

        return Conversation(turns, metadata=rollout, conv_type=conv_type)

    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return {
            "benign_train_samples": len(self.benign_train),
            "benign_valid_samples": len(self.benign_valid),
            "benign_test_samples": len(self.benign_test),
            "harmful_train_samples": len(self.harmful_train),
            "harmful_valid_samples": len(self.harmful_valid),
            "harmful_test_samples": len(self.harmful_test),
            "benign_train_rollouts": sum(
                len(rollouts) for rollouts in self.benign_train.values()
            ),
            "benign_valid_rollouts": sum(
                len(rollouts) for rollouts in self.benign_valid.values()
            ),
            "benign_test_rollouts": sum(
                len(rollouts) for rollouts in self.benign_test.values()
            ),
            "harmful_train_rollouts": sum(
                len(rollouts) for rollouts in self.harmful_train.values()
            ),
            "harmful_valid_rollouts": sum(
                len(rollouts) for rollouts in self.harmful_valid.values()
            ),
            "harmful_test_rollouts": sum(
                len(rollouts) for rollouts in self.harmful_test.values()
            ),
            "train_ratio": self.train_ratio,
            "valid_ratio": self.valid_ratio,
            "rollouts_per_sample": self.rollouts_per_sample,
        }

    def get_samples_raw(
        self, split: str = "train", conv_type: str = "benign"
    ) -> Dict[int, List[Dict]]:
        """
        Get samples with rollouts as raw dictionaries (for RL training).

        Args:
            split: 'train', 'valid', or 'test'
            conv_type: 'benign' or 'harmful'

        Returns:
            Dict mapping sample_index -> list of 20 rollout dictionaries

        Example:
            {
                0: [{rollout_1_dict}, {rollout_2_dict}, ..., {rollout_20_dict}],
                1: [{rollout_1_dict}, {rollout_2_dict}, ..., {rollout_20_dict}],
                ...
            }
        """
        if split == "train":
            samples = self.benign_train if conv_type == "benign" else self.harmful_train
        elif split == "valid":
            samples = self.benign_valid if conv_type == "benign" else self.harmful_valid
        else:
            samples = self.benign_test if conv_type == "benign" else self.harmful_test

        return samples
