"""社長音声タブ。

stt-desktop の CeoView を Streamlit 向けに移植したもの。
UI 構成は desktop に合わせている:
  - 参照フォルダ表示
  - 「未処理を自動取り込み」「音声ファイルを選択」ボタン
  - 「最新スキャン結果」4カラム（スキャン後のみ）
  - 「一括アップロード / 順次処理」キュー（常時表示）
  - 「処理結果」グリッド（常時表示、待機中は "-"）
"""

from __future__ import annotations

import html
import os
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
    CeoScanResult,
    CeoSourceFileEntry,
    process_ceo_path,
    process_ceo_uploaded_path,
    scan_ceo_source_directory,
)


DEFAULT_CEO_VAD_ENABLED = True
DEFAULT_CEO_VAD_AGGRESSIVENESS = 2


# ---------- state helpers ----------

def _ensure_state() -> dict:
    return st.session_state.setdefault(
        "ceo_tab_state",
        {
            "last_summary": None,   # CeoBatchSummary | None
            "last_scan": None,      # CeoScanResult | None
            "pending": None,        # dict | None — モーダルで取り込み待ちのファイル群
            "active_idx": 0,        # 「処理結果」で表示する結果のインデックス
            "uploader_key": 0,
        },
    )


def _ceo_source_dir() -> str:
    """参照フォルダの優先順: AppSettings (.app_settings.json) > 環境変数。"""

    app_settings = st.session_state.get("settings")
    if app_settings is not None:
        try:
            v = app_settings.get_ceo_source_dir()
            if v:
                return v
        except Exception:
            pass
    return os.getenv("CEO_SOURCE_DIR", "").strip()


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


# ---------- pending payload structure ----------
# pending = {
#   "source": "upload" | "scan",
#   "files": [
#       {
#         source_kind, file_path|None, temp_file_path|None, file_name,
#         size_bytes, modified_at|None, source_file_hash|None
#       }, ...
#   ],
# }


def _ceo_vad_settings() -> tuple[bool, int]:
    app_settings = st.session_state.get("settings")
    use_vad = bool(getattr(app_settings, "get_use_vad", lambda: DEFAULT_CEO_VAD_ENABLED)())
    vad_aggr = int(getattr(app_settings, "get_vad_aggressiveness", lambda: DEFAULT_CEO_VAD_AGGRESSIVENESS)())
    return use_vad, vad_aggr


def _persist_uploaded_file(uf) -> dict:
    temp_dir = Path(tempfile.mkdtemp(prefix="stt_ceo_upload_"))
    file_name = Path(uf.name).name or "uploaded_audio"
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
        "source_kind": "upload",
        "file_path": None,
        "temp_file_path": str(temp_path),
        "temp_dir": str(temp_dir),
        "file_name": file_name,
        "size_bytes": size,
        "modified_at": None,
        "source_file_hash": digest.hexdigest(),
    }


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


def _open_modal_for_upload(uploaded_files) -> None:
    state = _ensure_state()
    current = state.get("pending") or {}
    if current.get("source") == "upload":
        _cleanup_temp_uploads(current.get("files"))
    files = [_persist_uploaded_file(uf) for uf in uploaded_files]
    state["pending"] = {"source": "upload", "files": files}


def _open_modal_for_scan(entries: list[CeoSourceFileEntry]) -> None:
    state = _ensure_state()
    files = []
    for e in entries:
        files.append(
            {
                "source_kind": "scan",
                "file_path": e.file_path,
                "temp_file_path": None,
                "file_name": e.file_name,
                "size_bytes": e.size_bytes,
                "modified_at": e.modified_at,
                "source_file_hash": None,
            }
        )
    state["pending"] = {"source": "scan", "files": files}


# ---------- modal ----------

@st.dialog("メタ情報入力", width="large")
def _meta_input_modal(*, selected_model: str, logger) -> None:
    state = _ensure_state()
    pending = state.get("pending")
    if not pending:
        st.warning("対象ファイルがありません。")
        return

    is_scan = pending["source"] == "scan"
    st.subheader("自動検出ファイルの取り込み設定" if is_scan else "音声ファイルを選択")

    # upload モードの場合、モーダル内にファイル選択UIを出す
    if not is_scan:
        uploader_key = f"ceo_modal_uploader_{state['uploader_key']}"
        uploaded = st.file_uploader(
            "音声ファイルを選択",
            type=["wav", "mp3", "m4a", "flac", "ogg", "webm", "aac"],
            accept_multiple_files=True,
            key=uploader_key,
            label_visibility="collapsed",
        )
        if uploaded:
            # rerun のたびに pending["files"] を再構成するが、音声bytesは
            # session_stateに保持せず、一時ファイルのパスだけを残す。
            _cleanup_temp_uploads(pending.get("files"))
            pending["files"] = [_persist_uploaded_file(uf) for uf in uploaded]

    st.caption(f"対象ファイル数: {len(pending.get('files', []))} 件")

    title_input = st.text_input(
        "タイトル（空欄なら各ファイル名を使用）",
        value="",
        key="ceo_modal_title",
    )
    speaker_input = st.text_input(
        "話者",
        value=DEFAULT_CEO_SPEAKER,
        key="ceo_modal_speaker",
    )
    recorded_at_input = st.text_input(
        "録音日時（ISO 8601, 例: 2026-05-17T10:00:00。空欄ならファイル更新日時 → 現在時刻）",
        value="",
        key="ceo_modal_recorded_at",
    )

    if pending.get("files"):
        with st.expander(f"対象ファイル一覧（{len(pending['files'])}件）", expanded=False):
            for f in pending["files"][:50]:
                st.write(f"- {f['file_name']} ({(f['size_bytes'] or 0)/1024:.1f} KB)")
            if len(pending["files"]) > 50:
                st.caption(f"…ほか {len(pending['files']) - 50} 件")

    col_cancel, col_go = st.columns([1, 1])
    with col_cancel:
        if st.button("キャンセル", use_container_width=True, key="ceo_modal_cancel"):
            _cleanup_temp_uploads(pending.get("files"))
            state["pending"] = None
            st.rerun()
    with col_go:
        go_disabled = not pending.get("files")
        if st.button(
            "🚀 取り込み開始",
            use_container_width=True,
            type="primary",
            disabled=go_disabled,
            key="ceo_modal_go",
        ):
            speaker = (speaker_input or DEFAULT_CEO_SPEAKER).strip() or DEFAULT_CEO_SPEAKER
            title_override = title_input.strip()
            recorded_at_override = recorded_at_input.strip() or None

            summary = CeoBatchSummary()
            progress = st.progress(0.0)
            status = st.empty()
            total = len(pending["files"])
            use_vad, vad_aggressiveness = _ceo_vad_settings()

            try:
                for idx, f in enumerate(pending["files"]):
                    status.text(f"処理中: {f['file_name']} ({idx + 1}/{total})")
                    per_title = title_override or Path(f["file_name"]).stem
                    effective_recorded_at = (
                        recorded_at_override or f.get("modified_at") or datetime.now().isoformat(timespec="seconds")
                    )

                    if f["source_kind"] == "scan":
                        logger.info(
                            "CEO 処理開始 (scan): path=%s size=%s mtime=%s",
                            f["file_path"], f["size_bytes"], f["modified_at"],
                        )
                        result = process_ceo_path(
                            file_path=f["file_path"],
                            title=per_title,
                            speaker=speaker,
                            recorded_at=effective_recorded_at,
                            source_file_size_bytes=f["size_bytes"],
                            source_file_modified_at=f["modified_at"],
                            selected_model=selected_model,
                            use_vad=use_vad,
                            vad_aggressiveness=vad_aggressiveness,
                        )
                    else:
                        logger.info(
                            "CEO 処理開始 (upload): name=%s size=%s hash=%s",
                            f["file_name"], f["size_bytes"], f.get("source_file_hash"),
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
            finally:
                _cleanup_temp_uploads(pending.get("files"))

            status.text(
                f"完了: 成功 {summary.ok_count} / 重複スキップ {summary.skipped_count} / 失敗 {summary.error_count}"
            )

            state["last_summary"] = summary
            state["active_idx"] = 0
            state["pending"] = None
            if pending["source"] == "upload":
                state["uploader_key"] += 1
            st.rerun()


# ---------- scan result rendering ----------

def _render_file_list(title: str, files: list[CeoSourceFileEntry], empty_label: str) -> None:
    with st.container(border=True):
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(f"**{title}**")
        with c2:
            st.caption(f"{len(files)} 件")
        if not files:
            st.caption(empty_label)
            return
        for f in files[:8]:
            file_path = _html_text(f.file_path)
            file_name = _html_text(f.file_name)
            modified_at = _html_text(
                _format_datetime(f.modified_at) if f.modified_at else "更新日時なし"
            )
            st.markdown(
                f"<div style='font-size:12px;'>"
                f"<div style='overflow:hidden;text-overflow:ellipsis;white-space:nowrap;' title='{file_path}'>{file_name}</div>"
                f"<div style='color:#777;'>{modified_at}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        if len(files) > 8:
            st.caption(f"ほか {len(files) - 8} 件")


def _render_scan_summary(scan: CeoScanResult) -> None:
    with st.container(border=True):
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown("### 最新スキャン結果")
            st.caption(f"対象: {scan.directory_path}")
        with c2:
            st.caption(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        cols = st.columns(4)
        with cols[0]:
            _render_file_list("今回取り込み対象", scan.queued, "新規の未処理ファイルはありません。")
        with cols[1]:
            _render_file_list("既処理", scan.already_processed, "処理済みファイルはありません。")
        with cols[2]:
            _render_file_list("既にキュー済み", [], "キュー済みの重複ファイルはありません。")
        with cols[3]:
            _render_file_list("VAD生成物として除外", scan.skipped_generated, "除外された VAD ファイルはありません。")


# ---------- queue (一括アップロード / 順次処理) ----------

_STATUS_BADGE = {
    "ok": ("完了", "#16a34a"),
    "skipped_duplicate": ("重複", "#6b7280"),
    "error": ("失敗", "#dc2626"),
}


def _render_queue(summary: Optional[CeoBatchSummary]) -> None:
    """desktop の RecorderUploadQueue 相当。Streamlit は同期実行のため
    「処理中 / 待機」は常に 0 として表示し、完了後の集計のみ反映する。
    """

    done = err = 0
    rows: list[tuple[str, str, str, str]] = []  # (status_label, source_label, file_name, memo)
    if summary and summary.results:
        for r in summary.results:
            label, _ = _STATUS_BADGE.get(r.status, ("-", "#6b7280"))
            source = "自動検出" if r.source_kind == "scan" else "アップロード"
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
            st.markdown("### 一括アップロード / 順次処理")
        with c2:
            st.markdown(
                f"<div style='text-align:right;font-size:12px;color:#555;'>"
                f"処理中: 0 ・ 待機: 0 ・ 完了: {done} ・ 失敗: {err}"
                f"</div>",
                unsafe_allow_html=True,
            )
        st.caption("ファイル数に上限はありません。1件ずつ順番に処理します。")
        if not rows:
            st.caption("まだ処理中のファイルはありません。")
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


# ---------- main tab ----------

def run_ceo_tab(selected_model: str, logger) -> None:
    st.header("社長音声")
    st.caption(
        "参照フォルダの未処理音声を自動検出するか、手動で選択した音声を VAD 後に順次文字起こしします。"
    )

    state = _ensure_state()
    source_dir = _ceo_source_dir()

    # ----- 参照フォルダ表示 -----
    if source_dir:
        escaped_source_dir = _html_text(source_dir)
        st.markdown(
            f"<div style='font-size:12px;color:#444;margin-top:-4px;'><b>参照フォルダ:</b> {escaped_source_dir}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='font-size:12px;color:#8a6700;margin-top:-4px;'><b>参照フォルダ:</b> 未設定（サイドバーの『🎤 社長音声』で指定してください）</div>",
            unsafe_allow_html=True,
        )

    # ----- アクション行 -----
    col_scan, col_upload, _ = st.columns([1, 1, 3])

    with col_scan:
        if st.button(
            "未処理を自動取り込み",
            use_container_width=True,
            disabled=not source_dir,
            help="設定した参照フォルダから未処理音声を自動検出します",
            key="ceo_tab_scan",
        ):
            try:
                scan = scan_ceo_source_directory(source_dir)
                state["last_scan"] = scan
                if scan.queued:
                    _open_modal_for_scan(scan.queued)
                    st.rerun()
                else:
                    st.info("未処理の音声はありませんでした。")
            except Exception as exc:
                st.error(f"スキャンに失敗しました: {exc}")
                logger.exception("CEO scan failed")

    with col_upload:
        if st.button(
            "音声ファイルを選択",
            use_container_width=True,
            help="複数選択で順次処理します",
            key="ceo_tab_open_upload_modal",
        ):
            state["pending"] = {"source": "upload", "files": []}
            state["uploader_key"] += 1
            st.rerun()

    # ----- スキャン結果（スキャン後のみ）-----
    if state.get("last_scan") is not None:
        _render_scan_summary(state["last_scan"])

    # ----- 一括アップロード / 順次処理（常時表示）-----
    _render_queue(state.get("last_summary"))

    # ----- 処理結果（常時表示）-----
    _render_result_panel(state.get("last_summary"), state.get("active_idx", 0))

    # ----- メタ情報モーダル -----
    if state.get("pending"):
        _meta_input_modal(
            selected_model=selected_model or DEFAULT_CEO_MODEL,
            logger=logger,
        )
