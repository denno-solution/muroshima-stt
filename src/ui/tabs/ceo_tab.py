"""社長音声タブ。

Web版では、ブラウザのマイク録音を社長音声として `ceo_transcriptions` に保存する。
"""

from __future__ import annotations

import html
import tempfile
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Optional

import streamlit as st

from services.ceo_processor import (
    DEFAULT_CEO_MODEL,
    DEFAULT_CEO_SPEAKER,
    CeoBatchSummary,
    CeoProcessResult,
    process_ceo_uploaded_path,
)


DEFAULT_CEO_VAD_ENABLED = True
DEFAULT_CEO_VAD_AGGRESSIVENESS = 2


# ---------- state helpers ----------

def _ensure_state() -> dict:
    return st.session_state.setdefault(
        "ceo_tab_state",
        {
            "last_summary": None,   # CeoBatchSummary | None
            "active_idx": 0,        # 「処理結果」で表示する結果のインデックス
        },
    )


def _format_duration(value) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.1f} 秒"
    except Exception:
        return str(value)


def _format_datetime(value: Optional[str]) -> str:
    if not value:
        return "-"
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return value


def _html_text(value) -> str:
    return html.escape(str(value or ""), quote=True)


def _ceo_vad_settings() -> tuple[bool, int]:
    app_settings = st.session_state.get("settings")
    use_vad = bool(getattr(app_settings, "get_use_vad", lambda: DEFAULT_CEO_VAD_ENABLED)())
    vad_aggr = int(getattr(app_settings, "get_vad_aggressiveness", lambda: DEFAULT_CEO_VAD_AGGRESSIVENESS)())
    return use_vad, vad_aggr


def _suffix_from_audio_upload(uf, default: str = ".webm") -> str:
    name_suffix = Path(getattr(uf, "name", "") or "").suffix
    if name_suffix:
        return name_suffix
    content_type = (getattr(uf, "type", "") or "").lower()
    if "wav" in content_type:
        return ".wav"
    if "webm" in content_type:
        return ".webm"
    if "mpeg" in content_type or "mp3" in content_type:
        return ".mp3"
    if "ogg" in content_type:
        return ".ogg"
    return default


def _persist_uploaded_file(
    uf,
    *,
    source_kind: str = "mic",
    file_name_override: Optional[str] = None,
    temp_prefix: str = "stt_ceo_mic_",
) -> dict:
    temp_dir = Path(tempfile.mkdtemp(prefix=temp_prefix))
    file_name = Path(file_name_override or getattr(uf, "name", "") or "uploaded_audio").name
    suffix = Path(file_name).suffix or ".audio"
    temp_path = temp_dir / f"source{suffix}"
    digest = sha256()
    size = 0

    if hasattr(uf, "seek"):
        uf.seek(0)
    with open(temp_path, "wb") as out:
        while True:
            chunk = uf.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
            out.write(chunk)

    return {
        "source_kind": source_kind,
        "file_path": None,
        "temp_file_path": str(temp_path),
        "temp_dir": str(temp_dir),
        "file_name": file_name,
        "size_bytes": size,
        "modified_at": None,
        "source_file_hash": digest.hexdigest(),
    }


def _persist_mic_recording(audio_value) -> dict:
    timestamp = datetime.now()
    suffix = _suffix_from_audio_upload(audio_value)
    file_name = f"ceo_mic_{timestamp.strftime('%Y%m%d_%H%M%S')}{suffix}"
    entry = _persist_uploaded_file(
        audio_value,
        source_kind="mic",
        file_name_override=file_name,
        temp_prefix="stt_ceo_mic_",
    )
    entry["modified_at"] = timestamp.isoformat(timespec="seconds")
    entry["recorded_at"] = timestamp.isoformat(timespec="seconds")
    return entry


def _cleanup_temp_uploads(files) -> None:
    for f in files or []:
        temp_path = f.get("temp_file_path")
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass
        temp_dir = f.get("temp_dir")
        if temp_dir:
            try:
                Path(temp_dir).rmdir()
            except Exception:
                pass


def _process_ceo_entries(
    files: list[dict],
    *,
    title_override: str,
    speaker: str,
    recorded_at_override: Optional[str],
    selected_model: str,
    logger,
) -> CeoBatchSummary:
    summary = CeoBatchSummary()
    progress = st.progress(0.0)
    status = st.empty()
    total = len(files)
    use_vad, vad_aggressiveness = _ceo_vad_settings()

    if total == 0:
        return summary

    for idx, f in enumerate(files):
        status.text(f"処理中: {f['file_name']} ({idx + 1}/{total})")
        per_title = title_override or Path(f["file_name"]).stem
        effective_recorded_at = (
            recorded_at_override
            or f.get("recorded_at")
            or f.get("modified_at")
            or datetime.now().isoformat(timespec="seconds")
        )

        logger.info(
            "CEO 処理開始 (%s): name=%s size=%s hash=%s",
            f["source_kind"], f["file_name"], f["size_bytes"], f.get("source_file_hash"),
        )
        result = process_ceo_uploaded_path(
            file_name=f["file_name"],
            temp_file_path=f["temp_file_path"],
            title=per_title,
            speaker=speaker,
            recorded_at=effective_recorded_at,
            source_file_size_bytes=f["size_bytes"],
            source_file_modified_at=f.get("modified_at"),
            source_file_hash=f.get("source_file_hash"),
            selected_model=selected_model,
            use_vad=use_vad,
            vad_aggressiveness=vad_aggressiveness,
            cleanup_source=True,
        )
        summary.results.append(result)
        progress.progress((idx + 1) / total)

    status.text(
        f"完了: 成功 {summary.ok_count} / 重複スキップ {summary.skipped_count} / 失敗 {summary.error_count}"
    )
    return summary


# ---------- queue (マイク録音 / 処理履歴) ----------

_STATUS_BADGE = {
    "ok": ("完了", "#16a34a"),
    "skipped_duplicate": ("重複", "#6b7280"),
    "error": ("失敗", "#dc2626"),
}


def _source_label(source_kind: str) -> str:
    if source_kind == "mic":
        return "マイク録音"
    return "社長音声"


def _render_queue(summary: Optional[CeoBatchSummary]) -> None:
    """desktop の RecorderUploadQueue 相当。Streamlit は同期実行のため
    「処理中 / 待機」は常に 0 として表示し、完了後の集計のみ反映する。
    """

    done = err = 0
    rows: list[tuple[str, str, str, str]] = []  # (status_label, source_label, file_name, memo)
    if summary and summary.results:
        for r in summary.results:
            label, _ = _STATUS_BADGE.get(r.status, ("-", "#6b7280"))
            source = _source_label(r.source_kind)
            memo = ""
            if r.status == "error":
                memo = r.error or "処理に失敗しました"
            elif r.status == "skipped_duplicate":
                memo = f"既存ID: {r.matched_existing_id}" if r.matched_existing_id else "重複"
            elif r.status == "ok":
                memo = "完了"
            rows.append((label, source, r.file_name, memo))
        done = sum(1 for r in summary.results if r.status == "ok")
        err = sum(1 for r in summary.results if r.status == "error")

    with st.container(border=True):
        c1, c2 = st.columns([3, 4])
        with c1:
            st.markdown("### マイク録音 / 処理履歴")
        with c2:
            st.markdown(
                f"<div style='text-align:right;font-size:12px;color:#555;'>"
                f"処理中: 0 ・ 待機: 0 ・ 完了: {done} ・ 失敗: {err}"
                f"</div>",
                unsafe_allow_html=True,
            )
        st.caption("録音ごとに1件ずつ処理します。")
        if not rows:
            st.caption("まだ処理した録音はありません。")
            return
        # シンプルなテーブル風表示
        st.markdown(
            "<table style='width:100%;border-collapse:collapse;font-size:12px;'>"
            "<thead><tr style='background:#f4f6f8;'>"
            "<th style='text-align:left;padding:6px 8px;width:80px;'>状態</th>"
            "<th style='text-align:left;padding:6px 8px;width:100px;'>種別</th>"
            "<th style='text-align:left;padding:6px 8px;'>ファイル名</th>"
            "<th style='text-align:left;padding:6px 8px;'>メモ</th>"
            "</tr></thead><tbody>"
            + "".join(
                (
                    f"<tr style='border-bottom:1px solid #f0f2f5;'>"
                    f"<td style='padding:6px 8px;color:{_STATUS_BADGE.get(_status_key_from_label(label), ('','#555'))[1]};'>● {_html_text(label)}</td>"
                    f"<td style='padding:6px 8px;'>{_html_text(source)}</td>"
                    f"<td style='padding:6px 8px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:320px;' title='{_html_text(name)}'>{_html_text(name)}</td>"
                    f"<td style='padding:6px 8px;color:#555;'>{_html_text(memo)}</td>"
                    f"</tr>"
                )
                for (label, source, name, memo) in rows
            )
            + "</tbody></table>",
            unsafe_allow_html=True,
        )


def _status_key_from_label(label: str) -> str:
    for k, (status_label, _) in _STATUS_BADGE.items():
        if status_label == label:
            return k
    return "ok"


# ---------- 処理結果（常時表示） ----------

def _render_result_panel(summary: Optional[CeoBatchSummary], active_idx: int) -> None:
    active: Optional[CeoProcessResult] = None
    if summary and summary.results:
        active_idx = max(0, min(active_idx, len(summary.results) - 1))
        active = summary.results[active_idx]

    # ステータスバッジ
    if active is None:
        badge_text, badge_bg, badge_fg = "待機中", "#eee", "#666"
    elif active.status == "ok":
        badge_text, badge_bg, badge_fg = "完了", "#e8f8ef", "#1a7f4b"
    elif active.status == "skipped_duplicate":
        badge_text, badge_bg, badge_fg = "重複スキップ", "#f3f4f6", "#374151"
    else:
        badge_text, badge_bg, badge_fg = "失敗", "#fdecea", "#b00020"

    with st.container(border=True):
        head_l, head_r = st.columns([3, 1])
        with head_l:
            st.markdown("### 処理結果")
        with head_r:
            st.markdown(
                f"<div style='text-align:right;'><span style='font-size:12px;padding:2px 10px;border-radius:999px;background:{badge_bg};color:{badge_fg};'>{badge_text}</span></div>",
                unsafe_allow_html=True,
            )

        # ジョブ切替（結果が複数あるとき）
        if summary and len(summary.results) > 1:
            options = list(range(len(summary.results)))
            labels = [f"{i+1}. {r.file_name}" for i, r in enumerate(summary.results)]
            selected = st.selectbox(
                "結果を選択",
                options=options,
                index=active_idx,
                format_func=lambda i: labels[i],
                key="ceo_active_result_select",
            )
            if selected != active_idx:
                state = _ensure_state()
                state["active_idx"] = selected
                st.rerun()

        # 詳細グリッド
        cols = st.columns(3)
        with cols[0]:
            st.markdown("**ファイル**")
            st.write(active.file_name if active else "-")
            st.markdown("**タイトル**")
            st.write(active.title if active and active.title else "-")
            st.markdown("**DB保存**")
            if active is None:
                st.write("-")
            elif active.status == "ok":
                st.write(f"✅ 完了 (ID: {active.record_id})")
            elif active.status == "skipped_duplicate":
                st.write(f"⏭️ 既存 (ID: {active.matched_existing_id or '-'})")
            else:
                st.write("未保存")
        with cols[1]:
            st.markdown("**元音声**")
            st.write(active.file_name if active else "-")
            st.markdown("**話者**")
            st.write(active.speaker if active and active.speaker else "-")
        with cols[2]:
            st.markdown("**VAD保存先**")
            st.write(active.saved_path if active and active.saved_path else "-")
            st.markdown("**録音日時**")
            st.write(_format_datetime(active.recorded_at) if active else "-")
            st.markdown("**長さ**")
            st.write(_format_duration(active.duration_seconds) if active else "-")

        if active and active.vad_note:
            st.caption(active.vad_note)
        if active and active.warning:
            st.warning(active.warning)
        if active and active.error:
            st.error(active.error)

        st.markdown("**文字起こし**")
        st.text_area(
            "文字起こし結果",
            value=(active.transcript or "") if active else "",
            height=180,
            placeholder="ここに結果が表示されます",
            key=f"ceo_result_text_{active_idx if active else 'placeholder'}",
            label_visibility="collapsed",
        )


def _audio_value_digest(audio_value) -> Optional[str]:
    try:
        raw = audio_value.getvalue() if hasattr(audio_value, "getvalue") else audio_value
        return sha256(raw).hexdigest()
    except Exception:
        return None


def _render_mic_recorder(selected_model: str, logger) -> None:
    state = _ensure_state()

    with st.container(border=True):
        st.markdown("### マイクで録音して取り込み")
        st.caption("ブラウザのマイクで録音した音声を、社長音声として ceo_transcriptions に保存します。")

        col_title, col_speaker, col_time = st.columns([2, 1, 2])
        with col_title:
            title_input = st.text_input(
                "タイトル（空欄なら録音ファイル名を使用）",
                value="",
                key="ceo_mic_title",
            )
        with col_speaker:
            speaker_input = st.text_input(
                "話者",
                value=DEFAULT_CEO_SPEAKER,
                key="ceo_mic_speaker",
            )
        with col_time:
            recorded_at_input = st.text_input(
                "録音日時（空欄なら録音完了時刻）",
                value="",
                key="ceo_mic_recorded_at",
                help="ISO 8601形式で指定できます。例: 2026-05-21T10:00:00",
            )

        audio_value = st.audio_input(
            "🎙️ 社長音声を録音してください",
            help="録音を停止したあと、「社長音声として取り込む」を押してください。",
            key="ceo_mic_audio_input",
        )
        if not audio_value:
            return

        st.success("録音完了！")
        current_digest = _audio_value_digest(audio_value)
        if current_digest and st.session_state.get("ceo_mic_last_digest") == current_digest:
            st.caption("この録音は直近で取り込み済みです。再取り込みした場合は重複として扱われます。")

        disabled = bool(st.session_state.get("ceo_mic_processing", False))
        if not st.button(
            "社長音声として取り込む",
            type="primary",
            use_container_width=True,
            disabled=disabled,
            key="ceo_mic_process_button",
        ):
            return

        st.session_state.ceo_mic_processing = True
        files: list[dict] = []
        try:
            files = [_persist_mic_recording(audio_value)]
            speaker = (speaker_input or DEFAULT_CEO_SPEAKER).strip() or DEFAULT_CEO_SPEAKER
            title_override = title_input.strip()
            recorded_at_override = recorded_at_input.strip() or None

            with st.spinner("社長音声として文字起こし中..."):
                summary = _process_ceo_entries(
                    files,
                    title_override=title_override,
                    speaker=speaker,
                    recorded_at_override=recorded_at_override,
                    selected_model=selected_model or DEFAULT_CEO_MODEL,
                    logger=logger,
                )

            state["last_summary"] = summary
            state["active_idx"] = 0
            if current_digest:
                st.session_state.ceo_mic_last_digest = current_digest
            if summary.ok_count:
                st.success("社長音声として保存しました。")
            elif summary.skipped_count:
                st.info("同じ録音が既に保存済みだったため、重複としてスキップしました。")
            elif summary.error_count:
                st.error("社長音声の取り込みに失敗しました。詳細は下の処理結果を確認してください。")
        except Exception as exc:
            st.error(f"社長音声の取り込みに失敗しました: {exc}")
            logger.exception("CEO mic recording failed")
        finally:
            _cleanup_temp_uploads(files)
            st.session_state.ceo_mic_processing = False


# ---------- main tab ----------

def run_ceo_tab(selected_model: str, logger) -> None:
    st.header("社長音声")
    st.caption("ブラウザのマイク録音を、社長音声として VAD 後に文字起こしします。")

    state = _ensure_state()

    _render_mic_recorder(selected_model or DEFAULT_CEO_MODEL, logger)

    # ----- マイク録音 / 処理履歴（常時表示）-----
    _render_queue(state.get("last_summary"))

    # ----- 処理結果（常時表示）-----
    _render_result_panel(state.get("last_summary"), state.get("active_idx", 0))
