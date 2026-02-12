# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...

import torch
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.reward_score import (
    math_compute_score,
    r1v_compute_score,
    seg_compute_score,
    seg_strict_compute_score_qa,
    seg_strict_compute_score8_region_qa,
    seg_iterative_compute_score,   # <-- NEW import
    seg_iterative_compute_score_dense,
)


class CustomRewardManager:
    def __init__(self, tokenizer: PreTrainedTokenizer, num_examine: int, compute_score: str):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.mode = compute_score

        if compute_score == "math":
            self.compute_score = math_compute_score
        elif compute_score == "r1v":
            self.compute_score = r1v_compute_score
        elif compute_score == "seg":
            self.compute_score = seg_compute_score
        elif compute_score == "seg_strict_qa":
            self.compute_score = seg_strict_compute_score_qa
        elif compute_score == "seg_restrict8_regions_and_qa":
            self.compute_score = seg_strict_compute_score8_region_qa
        elif compute_score == "seg_iterative_qa":          # <-- NEW mode
            self.compute_score = seg_iterative_compute_score
        elif compute_score == "seg_iterative_qa_dense":    # <-- NEW mode
            self.compute_score = seg_iterative_compute_score_dense
        else:
            raise NotImplementedError()
        print("Using reward function:{}".format(self.compute_score))

    def __call__(self, data: DataProto, all_scores: bool = False) -> torch.Tensor:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        already_print = 0
        scores = None

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            if self.mode in ("seg_iterative_qa", "seg_iterative_qa_dense"):
                # --- Use fields directly from dataset for iterative reward ---
                nt = data_item.non_tensor_batch

                gt_answer_json = nt["answer"][0]          # GT JSON string
                input_bbox = nt.get("input_bbox", [None])[0]
                output_level = nt.get("output_level", ["medium"])[0]
                size_bucket = nt.get("size_bucket", [None])[0]

                score = self.compute_score(
                    response_str,
                    gt_answer_json,
                    input_bbox=input_bbox,
                    output_level=output_level,
                    size_bucket=size_bucket,
                )
            else:
                # --- Original strict seg reward path ---
                ground_truth = data_item.non_tensor_batch["solution"][0]
                score = self.compute_score(response_str, ground_truth)
                if isinstance(score, tuple) and len(score) == 2:
                    score, scores = score

            reward_tensor[i, valid_response_length - 1] = score

            if already_print < self.num_examine:
                already_print += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                if self.mode in ("seg_iterative_qa", "seg_iterative_qa_dense"):
                    print("[gt_answer_json]", gt_answer_json)
                    print("[input_bbox]", input_bbox)
                    print("[output_level]", output_level)
                    print("[size_bucket]", size_bucket)
                else:
                    print("[ground_truth]", ground_truth)
                print("[score]", score)

        if all_scores:
            return reward_tensor, scores
        else:
            return reward_tensor
