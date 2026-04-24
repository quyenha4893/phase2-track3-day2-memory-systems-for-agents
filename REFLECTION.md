# Reflection: Privacy, Limitations, and Trade-offs

## Memory nao giup nhieu nhat?

Long-term profile memory giup agent on dinh nhat trong cac tinh huong hoi lai thong tin sau nhieu turn, vi no tach bien facts quan trong khoi recent conversation buffer.

## Memory nao nhay cam nhat?

Long-term profile memory la phan nhay cam nhat, vi no de chua PII va preference lau dai nhu ten, noi song, allergy, thoi quen. Neu retrieve sai hoac bi luu qua muc can thiet thi rui ro privacy se cao hon short-term memory.

## Rui ro privacy / PII

- Allergy, city, thoi quen hoc tap va preference deu co the tro thanh PII hoac du lieu nhay cam theo context.
- Semantic memory neu index tai lieu co chua PII thi co the retrieve nham thong tin khong lien quan.
- Episodic memory co the luu outcome cua mot task chua thong tin rieng tu neu khong loc truoc khi save.

## Consent, deletion, TTL

- Nen chi save long-term facts sau khi co user consent ro rang.
- Can co deletion flow cho tung backend:
  - profile store;
  - episodic log;
  - semantic index neu tai lieu nguon co thong tin user.
- Nen co TTL cho memory dai han, dac biet voi episodic va profile facts tam thoi.

## Limitations ky thuat hien tai

- Semantic retrieval hien tai la keyword fallback, chua phan giai nghia tot nhu vector embedding.
- Fact extraction dang rule-based, nen de miss cac cau noi tu do.
- Chua co scoring router nang cao de xep hang memory theo importance/recency.
- Token budget dang tinh bang word count, khong phai token count that.
- Chua co persistence ra file cho profile va episodic memory sau moi lan chay.
