作業用の一時コードなどは`snippets`フォルダに保存します。.gitignoreされているので自由に使ってOK。

このリポジトリはDual Repo戦略を採用しています。
`origin`と`private`の2つの`remote`があります。`private/`から始まるブランチの`upstream`は`private`であり、それ以外にPushされてはいけません。
