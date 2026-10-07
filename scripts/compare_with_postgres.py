"""Compare fusion model estimates with Postgres planner estimates and true counts.

Usage:
  python scripts/compare_with_postgres.py --test-csv data/queries/test_queries_with_counts.csv --conn postgresql://postgres:postgres@localhost:5432/postgres
"""
import argparse
import json
import joblib
import os
import sys
import numpy as np
import pandas as pd
from urllib.parse import urlsplit, urlunsplit, quote

try:
    import psycopg2
except Exception:
    psycopg2 = None

from src.estimator.factorjoin import FactorJoin
from sklearn.metrics import mean_absolute_error


def extract_postgres_rows_from_plan(plan_json):
    # plan_json is a Python object (list/dict) returned by EXPLAIN (FORMAT JSON)
    # traverse and find the first numeric field whose key contains 'rows'
    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(k, str) and 'rows' in k.lower() and isinstance(v, (int, float)):
                    return float(v)
                res = walk(v)
                if res is not None:
                    return res
        elif isinstance(obj, list):
            for item in obj:
                res = walk(item)
                if res is not None:
                    return res
        return None

    return walk(plan_json)


def pg_estimate_for_query(conn, sql):
    cur = conn.cursor()
    try:
        try:
            cur.execute("EXPLAIN (FORMAT JSON) " + sql)
        except Exception as e:
            # could be undefined table or syntax error on user's PG instance
            return -1
        row = cur.fetchone()
        # row[0] is JSON text in newer psycopg2 versions, or already parsed
        plan = row[0]
        if isinstance(plan, str):
            plan = json.loads(plan)
        est = extract_postgres_rows_from_plan(plan)
        return est if est is not None else -1
    finally:
        cur.close()


def q_error(preds, truth):
    preds = np.maximum(preds, 1e-6)
    truth = np.maximum(truth, 1e-6)
    return np.maximum(preds / truth, truth / preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-csv', default='data/queries/test_queries_with_counts.csv')
    parser.add_argument('--model', default='models/fusion_model.joblib')
    parser.add_argument('--conn', default='postgresql://postgres:Sriram@IITM@localhost:5432/postgres')
    parser.add_argument('--pg-host', default=None, help='Postgres host (overrides --conn)')
    parser.add_argument('--pg-port', type=int, default=None, help='Postgres port (overrides --conn)')
    parser.add_argument('--pg-user', default=None, help='Postgres user (overrides --conn)')
    parser.add_argument('--pg-password', default=None, help='Postgres password (overrides --conn)')
    parser.add_argument('--pg-db', default=None, help='Postgres database name (overrides --conn)')
    args = parser.parse_args()

    if psycopg2 is None:
        print('psycopg2 not available in this environment. Install psycopg2-binary and retry.')
        sys.exit(1)

    if not os.path.exists(args.test_csv):
        print('Test CSV not found:', args.test_csv)
        sys.exit(1)

    df = pd.read_csv(args.test_csv)

    # load model
    model = joblib.load(args.model)
    fj = FactorJoin(csv_dir='data/out')

    # connect to Postgres
    try:
        # If user provided explicit pg params, prefer those (avoids DSN encoding issues)
        if args.pg_host or args.pg_user or args.pg_password or args.pg_db:
            conn_kwargs = {}
            if args.pg_host:
                conn_kwargs['host'] = args.pg_host
            if args.pg_port:
                conn_kwargs['port'] = args.pg_port
            if args.pg_user:
                conn_kwargs['user'] = args.pg_user
            if args.pg_password:
                conn_kwargs['password'] = args.pg_password
            if args.pg_db:
                conn_kwargs['dbname'] = args.pg_db
            conn = psycopg2.connect(**conn_kwargs)
        else:
            # normalize connection string to URL-encode password if it contains special chars
            def normalize_conn(conn_str):
                if '://' not in conn_str:
                    return conn_str
                scheme, rest = conn_str.split('://', 1)
                if '@' not in rest:
                    return conn_str
                last_at = rest.rfind('@')
                creds = rest[:last_at]
                host_and_path = rest[last_at+1:]
                if ':' not in creds:
                    return conn_str
                user, pw = creds.split(':', 1)
                pw_enc = quote(pw, safe='')
                new_rest = f"{user}:{pw_enc}@{host_and_path}"
                return f"{scheme}://{new_rest}"

            conn = psycopg2.connect(normalize_conn(args.conn))
    except Exception as e:
        print('Failed to connect to Postgres with conn string or params')
        print('Conn args:', {k: (v if k != 'password' else '***') for k, v in ({'conn': args.conn} if not args.pg_host else conn_kwargs).items()})
        print('Error:', e)
        sys.exit(1)

    pg_ests = []
    model_preds = []
    truths = []

    for sql, true in zip(df['sql'].tolist(), df['true_count'].tolist()):
        est_pg = pg_estimate_for_query(conn, sql)
        feats = fj.extract_features(sql)
        X = np.array([[feats['s_e'], feats['s_d'], feats['s_p'], feats['p_dept_given_emp'], feats['p_proj_given_emp'], feats['factorjoin_est']]])
        pred_log = model.predict(X)[0]
        pred = float(np.expm1(pred_log))
        pg_ests.append(float(est_pg) if est_pg is not None else -1)
        model_preds.append(pred)
        truths.append(float(true))

    conn.close()

    truths = np.array(truths)
    model_preds = np.array(model_preds)
    pg_ests = np.array(pg_ests)

    # filter rows where postgres returned -1 (unknown)
    mask_pg_valid = pg_ests >= 0
    if mask_pg_valid.sum() > 0:
        q_pg = q_error(pg_ests[mask_pg_valid], truths[mask_pg_valid])
        med_q_pg = np.median(q_pg)
        mae_pg = mean_absolute_error(truths[mask_pg_valid], pg_ests[mask_pg_valid])
    else:
        med_q_pg = float('nan')
        mae_pg = float('nan')

    q_model = q_error(model_preds, truths)
    med_q_model = np.median(q_model)
    mae_model = mean_absolute_error(truths, model_preds)

    print('Postgres: Median Q-error =', med_q_pg, ', MAE =', mae_pg, ', valid_queries =', int(mask_pg_valid.sum()))
    print('Model   : Median Q-error =', med_q_model, ', MAE =', mae_model, ', queries =', len(truths))


if __name__ == '__main__':
    main()
