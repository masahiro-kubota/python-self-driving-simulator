#!/usr/bin/env python3
"""experiment-runnerのプロファイリングヘルパースクリプト

py-spyを使ってexperiment-runnerの実行時間を計測し、
flamegraphまたはSpeedscope形式で出力します。
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="experiment-runnerをプロファイリング実行します")
    parser.add_argument(
        "--format",
        choices=["flamegraph", "speedscope"],
        default="speedscope",
        help="出力形式 (default: speedscope)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="出力ファイル名 (default: profile.{svg|speedscope.json})",
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=100,
        help="サンプリングレート (Hz) (default: 100)",
    )
    parser.add_argument(
        "--subprocesses",
        action="store_true",
        help="サブプロセスも含めてプロファイリング",
    )
    parser.add_argument(
        "--native",
        action="store_true",
        help="C/C++拡張も含める (要root権限)",
    )

    args = parser.parse_args()

    # 出力ファイル名を決定
    if args.output:
        output_file = args.output
    else:
        if args.format == "speedscope":
            output_file = Path("profile.speedscope.json")
        else:
            output_file = Path("profile_flamegraph.svg")

    # py-spyコマンドを構築
    # uv run経由だとpy-spyがPythonプロセスを見つけられないため、
    # Pythonスクリプトを直接実行する
    cmd = [
        "py-spy",
        "record",
        "-o",
        str(output_file),
        "--rate",
        str(args.rate),
    ]

    if args.format == "speedscope":
        cmd.extend(["--format", "speedscope"])

    if args.subprocesses:
        cmd.append("--subprocesses")

    if args.native:
        cmd.append("--native")

    # experiment-runnerのエントリーポイントを直接実行
    # プロファイリング用に実行時間を短縮（1秒のみ）
    cmd.extend(
        [
            "--",
            "python",
            "-m",
            "experiment.cli",
            "execution.duration_sec=1",  # 1秒のみ実行
            "postprocess.dashboard.enabled=false",  # ダッシュボード生成を無効化
            "postprocess.mcap.enabled=false",  # MCAP出力を無効化
        ]
    )

    print(f"🔍 プロファイリング開始: {' '.join(cmd)}")
    print(f"📊 出力ファイル: {output_file.absolute()}")

    # テキスト形式のサマリーファイルも生成
    summary_file = Path("profile_summary.txt")
    print(f"📄 サマリーファイル: {summary_file.absolute()}")
    print()

    try:
        subprocess.run(cmd, check=False)  # check=Falseに変更してエラーを無視
        print()

        # プロファイルファイルが生成されていれば成功とみなす
        if output_file.exists():
            print("✅ プロファイリング完了!")
            print(f"📁 結果: {output_file.absolute()}")

            # テキスト形式のサマリーを生成（AIが読みやすい形式）
            print()
            print("📊 テキストサマリーを生成中...")
            try:
                # cProfileでプロファイリングしてテキスト形式で出力
                profile_cmd = [
                    "python",
                    "-m",
                    "cProfile",
                    "-o",
                    "profile.prof",
                    "-m",
                    "experiment.cli",
                    "execution.duration_sec=1",
                    "postprocess.dashboard.enabled=false",
                    "postprocess.mcap.enabled=false",
                ]

                subprocess.run(
                    profile_cmd,
                    capture_output=True,
                    timeout=30,
                    check=False,
                )

                # pstatsでテキスト形式に変換 + JSON/HTML生成
                import json
                import pstats
                import re
                from datetime import datetime
                from io import StringIO

                stats = pstats.Stats("profile.prof")
                stats.strip_dirs()

                # JSON用のデータを抽出
                profile_data = {
                    "timestamp": datetime.now().isoformat(),
                    "total_runtime": 0,
                    "total_calls": 0,
                    "top_bottleneck": "",
                    "cumulative": [],
                    "by_time": [],
                    "by_calls": [],
                }

                # 統計情報を抽出する関数
                def extract_stats(sort_key, limit=100):
                    stats.sort_stats(sort_key)
                    s = StringIO()
                    stats.stream = s
                    stats.print_stats(limit)
                    output = s.getvalue()

                    # パース
                    lines = output.split("\n")
                    data = []
                    total_time = 0
                    total_calls = 0

                    for line in lines:
                        # 統計行をパース
                        match = re.match(
                            r"\s*(\d+(?:/\d+)?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(.+)",
                            line,
                        )
                        if match:
                            ncalls_str, tottime, percall_tot, cumtime, percall_cum, func = (
                                match.groups()
                            )

                            # ncallsをパース（"100/50"のような形式にも対応）
                            ncalls = int(ncalls_str.split("/")[0])

                            data.append(
                                {
                                    "function": func.strip(),
                                    "ncalls": ncalls,
                                    "tottime": float(tottime),
                                    "cumtime": float(cumtime),
                                    "percall": float(percall_cum),
                                }
                            )

                            total_time = max(total_time, float(cumtime))
                            total_calls += ncalls

                    return data, total_time, total_calls

                # 各ソート順でデータを抽出
                cumulative_data, total_runtime, total_calls = extract_stats("cumulative", 100)
                time_data, _, _ = extract_stats("time", 100)
                calls_data, _, _ = extract_stats("calls", 50)

                profile_data["cumulative"] = cumulative_data
                profile_data["by_time"] = time_data
                profile_data["by_calls"] = calls_data
                profile_data["total_runtime"] = total_runtime
                profile_data["total_calls"] = total_calls
                profile_data["top_bottleneck"] = (
                    cumulative_data[0]["function"] if cumulative_data else "N/A"
                )

                # JSONファイルを生成
                json_file = Path("profile_data.json")
                with open(json_file, "w") as f:
                    json.dump(profile_data, f, indent=2)

                print(f"✅ JSON生成完了: {json_file.absolute()}")

                # HTMLダッシュボードを生成
                template_path = Path(__file__).parent.parent / "profile_dashboard_template.html"
                dashboard_path = Path("profile_dashboard.html")

                if template_path.exists():
                    with open(template_path) as f:
                        template = f.read()

                    # データを注入（スペースの有無にかかわらず置換）
                    import re

                    html_content = re.sub(
                        r"\{\{\s*PROFILE_DATA\s*\}\}", json.dumps(profile_data, indent=2), template
                    )

                    with open(dashboard_path, "w") as f:
                        f.write(html_content)

                    print(f"✅ HTMLダッシュボード生成完了: {dashboard_path.absolute()}")
                else:
                    print(f"⚠️  テンプレートが見つかりません: {template_path}")

                # テキストサマリーも生成
                stats_for_text = pstats.Stats("profile.prof")
                stats_for_text.strip_dirs()

                # サマリーをファイルに保存
                with open(summary_file, "w") as f:
                    f.write("# experiment-runner プロファイリングサマリー\\n\\n")

                    # 1. 累積時間順（上位100件）
                    f.write("## 1. 累積時間順（上位100件）\\n\\n")
                    f.write("関数が直接・間接的に消費した総時間。ボトルネック特定に最適。\\n\\n")
                    f.write("```\\n")
                    s = StringIO()
                    stats_for_text.stream = s
                    stats_for_text.sort_stats("cumulative")
                    stats_for_text.print_stats(100)
                    f.write(s.getvalue())
                    f.write("```\\n\\n")

                    # 2. 実行時間順（上位100件）
                    f.write("## 2. 実行時間順（上位100件）\\n\\n")
                    f.write("関数自体の実行時間（サブ関数を除く）。最適化対象の特定に最適。\\n\\n")
                    f.write("```\\n")
                    s = StringIO()
                    stats_for_text.stream = s
                    stats_for_text.sort_stats("time")
                    stats_for_text.print_stats(100)
                    f.write(s.getvalue())
                    f.write("```\\n\\n")

                    # 3. 呼び出し回数順（上位50件）
                    f.write("## 3. 呼び出し回数順（上位50件）\\n\\n")
                    f.write("頻繁に呼ばれる関数。キャッシュや最適化の候補。\\n\\n")
                    f.write("```\\n")
                    s = StringIO()
                    stats_for_text.stream = s
                    stats_for_text.sort_stats("calls")
                    stats_for_text.print_stats(50)
                    f.write(s.getvalue())
                    f.write("```\\n\\n")

                    # 4. 呼び出し元情報（上位30件）
                    f.write("## 4. 呼び出し元情報（累積時間順、上位30件）\\n\\n")
                    f.write("どの関数から呼ばれているかを確認。\\n\\n")
                    f.write("```\\n")
                    s = StringIO()
                    stats_for_text.stream = s
                    stats_for_text.sort_stats("cumulative")
                    stats_for_text.print_callers(30)
                    f.write(s.getvalue())
                    f.write("```\\n\\n")

                    # 5. 呼び出し先情報（上位30件）
                    f.write("## 5. 呼び出し先情報（累積時間順、上位30件）\\n\\n")
                    f.write("どの関数を呼んでいるかを確認。\\n\\n")
                    f.write("```\\n")
                    s = StringIO()
                    stats_for_text.stream = s
                    stats_for_text.sort_stats("cumulative")
                    stats_for_text.print_callees(30)
                    f.write(s.getvalue())
                    f.write("```\\n")

                # 一時ファイルを削除
                Path("profile.prof").unlink(missing_ok=True)

                print(f"✅ サマリー生成完了: {summary_file.absolute()}")
            except Exception as e:
                print(f"⚠️  サマリー生成に失敗: {e}")
                import traceback

                traceback.print_exc()

            if args.format == "speedscope":
                print()
                print("🌐 Speedscopeで表示:")
                print("   1. https://www.speedscope.app/ を開く")
                print(f"   2. {output_file.name} をドラッグ&ドロップ")
            else:
                print()
                print("🌐 Flamegraphを表示:")
                print(f"   ブラウザで {output_file.absolute()} を開いてください")

            print()
            print(f"💡 AIが読める形式: {summary_file.absolute()}")

            return 0  # 成功
        else:
            print("❌ エラー: プロファイルファイルが生成されませんでした")
            return 1

    except KeyboardInterrupt:
        print("\n⚠️  中断されました")
        return 130


if __name__ == "__main__":
    sys.exit(main())
